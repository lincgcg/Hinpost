"""Training, checkpoint transfer, and independent evaluation."""
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
import torch
from uer.opts import model_opts
from uer.utils import str2optimizer, str2scheduler, str2tokenizer
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from hinpost.model import Hinpost
from hinpost.data.dataset import read_dataset, batches
from hinpost.data.prepare import sha256

def load_or_initialize_parameters(args, model):
    """Preserve the supplied stage-transfer rule and report checkpoint coverage."""
    if args.pretrained_model_path:
        state = torch.load(args.pretrained_model_path, map_location="cpu", weights_only=True)
        incompatible = model.load_state_dict(state, strict=False)
        print(json.dumps({"missing_keys": incompatible.missing_keys,
                          "unexpected_keys": incompatible.unexpected_keys}, indent=2))
        missing_backbone = [k for k in incompatible.missing_keys if k.startswith(("embedding.", "encoder."))]
        if missing_backbone:
            raise ValueError("Checkpoint does not fully cover the configured backbone; check the model configuration.")
        if args.evaluate_only and incompatible.missing_keys:
            raise ValueError("Evaluation requires every parameter of the selected analyzer.")
    else:
        for name, parameter in model.named_parameters():
            if "gamma" not in name and "beta" not in name:
                parameter.data.normal_(0, 0.02)
    if not args.evaluate_only:
        # The supplied code resets the auxiliary head/projection at every stage.
        for name, parameter in model.named_parameters():
            if "raw_moe" not in name and ("output_layer" in name or "moe" in name):
                parameter.data.normal_(0, 0.02)


def build_optimizer(args, model):
    no_decay = ("bias", "gamma", "beta")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    if args.optimizer == "adamw":
        optimizer = str2optimizer[args.optimizer](groups, lr=args.learning_rate, correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](groups, lr=args.learning_rate,
                                                scale_parameter=False, relative_step=False)
    if args.scheduler == "constant":
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler == "constant_with_warmup":
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps * args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps * args.warmup, args.train_steps)
    return optimizer, scheduler


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate(args, model, dataset, predictions_path=None):
    model.eval()
    prediction_rows = []
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)
    for src, labels, seg, _ in batches(dataset, args.batch_size, args.device):
        with torch.no_grad():
            _, logits = model(src, None, seg)
        predictions = logits.argmax(dim=1)
        if predictions_path:
            confidence = logits.softmax(dim=1).max(dim=1).values
            prediction_rows.extend(zip(labels.cpu().tolist(), predictions.cpu().tolist(), confidence.cpu().tolist()))
        # Rows are predictions, columns are ground truth, matching the supplied code.
        indices = (predictions * args.labels_num + labels).cpu()
        confusion += torch.bincount(indices, minlength=args.labels_num ** 2).reshape(confusion.shape)
    if predictions_path:
        destination = Path(predictions_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["row_index", "true_label", "predicted_label", "confidence"])
            writer.writerows((i, *row) for i, row in enumerate(prediction_rows))
    matrix = confusion.double()
    precision = matrix.diag() / matrix.sum(dim=1).clamp_min(1)
    recall = matrix.diag() / matrix.sum(dim=0).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {"accuracy": float(matrix.diag().sum() / matrix.sum()),
            "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
            "macro_f1": float(f1.mean()), "samples": len(dataset),
            "confusion_rows": "predicted", "confusion_columns": "true",
            "confusion_matrix": confusion.tolist()}


def main():
    parser = argparse.ArgumentParser(description="Train one Hinpost stage or evaluate a checkpoint.")
    defaults = argparse.ArgumentParser(add_help=False)
    model_opts(defaults)
    parser.set_defaults(**vars(defaults.parse_args([])))
    parser.add_argument("--config_path", default="configs/backbones/base_config.json")
    parser.add_argument("--vocab_path", default="models/encryptd_vocab.txt")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--pretrained_model_path")
    source.add_argument("--from_scratch", action="store_true", help="Explicit random initialization for validation.")
    parser.add_argument("--train_path")
    parser.add_argument("--dev_path")
    parser.add_argument("--test_path")
    parser.add_argument("--output_model_path", default="outputs/model.bin")
    parser.add_argument("--evaluate_output_path", default="outputs/metrics.json")
    parser.add_argument("--predictions_path", help="Write final test predictions in TSV row order.")
    parser.add_argument("--labels_num", type=int, required=True)
    parser.add_argument("--expert_num", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--soft_targets", action="store_true", help="Use paired-view post-training.")
    parser.add_argument("--soft_alpha", type=float, default=0.1)
    parser.add_argument("--soft_beta", type=float, default=20)
    parser.add_argument("--epochs_num", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_length", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--optimizer", choices=["adamw", "adafactor"], default="adamw")
    parser.add_argument("--scheduler", choices=["linear", "constant", "constant_with_warmup", "cosine"], default="linear")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:0 (visible device index).")
    parser.add_argument("--cpu_threads", type=int, default=0)
    parser.add_argument("--evaluate_only", action="store_true")
    parser.set_defaults(pooling2="first", tokenizer="bert", do_lower_case="true", merges_path=None, spm_model_path=None)
    args = load_hyperparam(parser.parse_args())
    resolved_configuration = vars(args).copy()
    if args.batch_size < 1 or args.labels_num < 2 or args.seq_length < 2 or args.seq_length > args.max_seq_length:
        parser.error("Require positive batch size, at least two classes, and 2 <= seq_length <= max_seq_length.")
    if not 1 <= args.top_k <= args.expert_num or args.soft_alpha < 0 or args.soft_beta < 0:
        parser.error("Invalid expert counts or negative loss weights.")
    if args.evaluate_only:
        if not args.pretrained_model_path or not args.test_path:
            parser.error("Evaluation requires pretrained_model_path and test_path.")
    elif not args.train_path or not args.dev_path or args.epochs_num < 1:
        parser.error("Training requires train_path, dev_path, and epochs_num >= 1.")
    if not args.pretrained_model_path and not args.from_scratch:
        parser.error("Provide pretrained_model_path or explicitly select from_scratch.")
    if not 0 <= args.warmup <= 1 or args.learning_rate <= 0:
        parser.error("Require warmup in [0, 1] and a positive learning rate.")
    if args.predictions_path and not args.test_path:
        parser.error("predictions_path requires test_path.")
    for target in [args.evaluate_output_path, args.predictions_path]:
        if target and Path(target).exists():
            parser.error(f"Output exists; choose a fresh path: {target}")
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
    set_seed(args.seed)
    args.tokenizer = str2tokenizer[args.tokenizer](args)
    model = Hinpost(args)
    load_or_initialize_parameters(args, model)
    args.device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    model = model.to(args.device)
    if args.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)
    Path(args.evaluate_output_path).parent.mkdir(parents=True, exist_ok=True)
    if args.evaluate_only:
        result = evaluate(args, model, read_dataset(args, args.test_path), args.predictions_path)
    else:
        output = Path(args.output_model_path)
        if output.exists():
            parser.error("output_model_path already exists; choose a new path for this run.")
        output.parent.mkdir(parents=True, exist_ok=True)
        train_set = read_dataset(args, args.train_path, is_training=True)
        dev_set = read_dataset(args, args.dev_path)
        random.shuffle(train_set)  # Retain the supplied one-time training permutation.
        args.train_steps = math.ceil(len(train_set) / args.batch_size) * args.epochs_num
        optimizer, scheduler = build_optimizer(args, model)
        best_accuracy, updates, history = -1.0, 0, []
        synchronize(args.device)
        wall_start = time.perf_counter()
        for epoch in range(1, args.epochs_num + 1):
            model.train()
            loss_sum = 0.0
            for src, labels, seg, augmented in batches(train_set, args.batch_size, args.device):
                optimizer.zero_grad()
                loss, _ = model(src, labels, seg, augmented)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss.")
                loss.backward()
                optimizer.step()
                scheduler.step()
                loss_sum += loss.item()
                updates += 1
            validation = evaluate(args, model, dev_set)
            if validation["accuracy"] > best_accuracy:
                best_accuracy = validation["accuracy"]
                save_model(model, str(output))
            history.append({"epoch": epoch, "mean_batch_loss": loss_sum / math.ceil(len(train_set) / args.batch_size),
                            "validation": validation})
            print(json.dumps(history[-1]))
        synchronize(args.device)
        elapsed = time.perf_counter() - wall_start
        model.load_state_dict(torch.load(str(output), map_location=args.device, weights_only=True))
        result = {"best_validation_accuracy": best_accuracy, "updates": updates,
                  "stage_wall_seconds": elapsed,
                  "timing_scope": "training, validation and checkpoint writes; excludes setup, data preparation and final test",
                  "checkpoint_selection": "validation accuracy", "history": history}
        if args.test_path:
            result["test"] = evaluate(args, model, read_dataset(args, args.test_path), args.predictions_path)
    result["resources"] = {
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(args.device) if args.device.type == "cuda" else None,
        "gpu_name": torch.cuda.get_device_name(args.device) if args.device.type == "cuda" else None,
    }
    result["implementation"] = "gated-sum experts with normalized projection L1"
    result["device"] = str(args.device)
    result["seed"] = args.seed
    result["torch_version"] = torch.__version__
    result["configuration"] = resolved_configuration
    result["input_sha256"] = {
        name: sha256(path) for name, path in
        ((name, getattr(args, name)) for name in
         ("config_path", "vocab_path", "pretrained_model_path", "train_path", "dev_path", "test_path"))
        if path is not None and (not args.evaluate_only or name not in {"train_path", "dev_path"})
    }
    if not args.evaluate_only:
        result["output_checkpoint_sha256"] = sha256(args.output_model_path)
    Path(args.evaluate_output_path).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
