"""Configuration-driven preparation and four-stage post-training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from hinpost.data.prepare import sha256

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("field-init", "field-shuffle", "packet-shuffle", "ip")


def load_config(path):
    config = json.loads(Path(path).read_text())
    required = {"schema_version", "name", "backbone", "vocabulary", "data", "training", "stages"}
    if set(config) != required or config["schema_version"] != 1:
        raise ValueError(f"Configuration must contain exactly {sorted(required)} and schema_version=1.")
    if tuple(s["method"] for s in config["stages"]) != STAGES:
        raise ValueError(f"The supplied curriculum order is {STAGES}.")
    if any(set(s) != {"method", "epochs", "consistency_weight"} or s["epochs"] < 1 or s["consistency_weight"] < 0 for s in config["stages"]):
        raise ValueError("Each stage requires positive epochs and a nonnegative consistency weight.")
    if config["stages"][-1]["consistency_weight"] != 0:
        raise ValueError("The original-only final stage must have zero consistency weight.")
    training = config["training"]
    expected = {"expert_num", "top_k", "seq_length", "batch_size", "learning_rate", "warmup", "soft_alpha", "seeds"}
    if set(training) != expected:
        raise ValueError(f"training keys must be {sorted(expected)}.")
    if not 1 <= training["top_k"] <= training["expert_num"]:
        raise ValueError("Require 1 <= top_k <= expert_num.")
    if not training["seeds"] or len(set(training["seeds"])) != len(training["seeds"]):
        raise ValueError("Training seeds must be nonempty and unique.")
    if set(config["data"]) != {"seed", "samples_per_class", "min_packet_bytes", "field_init_rate", "field_shuffle_rate"}:
        raise ValueError("Unexpected data settings; see configs/hinpost.json.")
    for name in ("backbone", "vocabulary"):
        asset = ROOT / config[name]
        if not asset.is_file():
            raise ValueError(f"Missing {name}: {asset}")
    return config


def validate_prepared(data_root, config=None):
    """Verify stage identity, label mapping, shared splits, and TSV hashes."""
    reference, inputs = None, {}
    for stage in STAGES:
        folder = Path(data_root) / stage
        manifest = json.loads((folder / "split_manifest.json").read_text())
        if reference is None:
            reference = manifest
        if manifest != reference:
            raise ValueError("All stages must share the same flow/split manifest.")
        labels = sorted(manifest["labels"].values())
        if labels != list(range(len(labels))) or len(labels) < 2:
            raise ValueError("Labels must be contiguous integers starting at zero, with at least two classes.")
        metadata = json.loads((folder / "dataset_metadata.json").read_text())
        if metadata["method"] != stage:
            raise ValueError(f"Wrong preparation method in {stage}.")
        if config is not None:
            expected = {"random_seed": config["data"]["seed"], "min_plen": config["data"]["min_packet_bytes"], "aug_time": 1}
            if stage in {"field-init", "field-shuffle"}:
                expected["selected_rate"] = config["data"][stage.replace("-", "_") + "_rate"]
            if any(metadata.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Prepared {stage} parameters differ from the experiment configuration.")
        for split in ("train", "valid", "test"):
            path = folder / f"{split}_dataset.tsv"
            digest = sha256(path)
            if digest != metadata["files"][path.name]:
                raise ValueError(f"Dataset changed: {stage}/{path.name}")
            inputs[f"{stage}/{path.name}"] = digest
    inputs["split_manifest"] = sha256(Path(data_root) / STAGES[0] / "split_manifest.json")
    return reference["labels"], inputs


def execute(command, log_path, cwd=ROOT):
    """Retain complete logs and surface a readable failure location."""
    print("Running: " + " ".join(command), flush=True)
    with Path(log_path).open("w") as stream:
        result = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONUNBUFFERED": "1"})
    if result.returncode:
        raise RuntimeError(f"Command exited with {result.returncode}; inspect {log_path}.")


def prepare_stages(config, pcap_root, data_root):
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=False)
    options = config["data"]
    for index, method in enumerate(STAGES):
        command = [sys.executable, "-m", "hinpost.data.prepare", "--method", method,
                   "--pcap_path", str(Path(pcap_root).resolve()), "--dataset_dir", str(data_root / method),
                   "--random_seed", str(options["seed"]), "--samples_per_class", str(options["samples_per_class"]),
                   "--min_plen", str(options["min_packet_bytes"])]
        if method in {"field-init", "field-shuffle"}:
            command += ["--selected_rate", str(options[method.replace("-", "_") + "_rate"])]
        if index:
            command += ["--split_manifest", str(data_root / STAGES[0] / "split_manifest.json")]
        execute(command, data_root / f"{method}.log")
    validate_prepared(data_root, config)


def train_seed(config, data_root, output, seed, device, cpu_threads, pretrained):
    labels, inputs = validate_prepared(data_root, config)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")
    record = {"status": "running", "seed": seed, "device": device, "configuration": config,
              "input_sha256": inputs, "asset_sha256": {k: sha256(ROOT / config[k]) for k in ("backbone", "vocabulary")},
              "initial_checkpoint_sha256": sha256(pretrained) if pretrained else None,
              "initialization": "provided_checkpoint" if pretrained else "random",
              "commands": [], "completed_stages": []}
    manifest = output / "run.json"
    def save():
        manifest.write_text(json.dumps(record, indent=2) + "\n")
    previous = pretrained
    start = time.perf_counter()
    try:
        for stage in config["stages"]:
            method = stage["method"]
            folder = data_root / method
            checkpoint = output / f"{method}.bin"
            command = [sys.executable, "-m", "hinpost.engine", "--config_path", str(ROOT / config["backbone"]),
                       "--vocab_path", str(ROOT / config["vocabulary"]), "--labels_num", str(len(labels)),
                       "--train_path", str(folder / "train_dataset.tsv"), "--dev_path", str(folder / "valid_dataset.tsv"),
                       "--output_model_path", str(checkpoint), "--evaluate_output_path", str(output / f"{method}.json"),
                       "--epochs_num", str(stage["epochs"]), "--soft_beta", str(stage["consistency_weight"]),
                       "--seed", str(seed), "--device", device, "--cpu_threads", str(cpu_threads)]
            for key, value in config["training"].items():
                if key != "seeds":
                    command += ["--" + key, str(value)]
            command += ["--pretrained_model_path", str(previous)] if previous else ["--from_scratch"]
            if method != "ip":
                command += ["--soft_targets"]
            else:
                command += ["--test_path", str(folder / "test_dataset.tsv"), "--predictions_path", str(output / "predictions.csv")]
            record["commands"].append(command)
            save()
            execute(command, output / f"{method}.log")
            record["completed_stages"].append(method)
            previous = checkpoint
        record.update(status="completed", elapsed_seconds=time.perf_counter() - start,
                      checkpoint_sha256={s: sha256(output / f"{s}.bin") for s in STAGES})
    except Exception as error:
        record.update(status="failed", error=str(error), elapsed_seconds=time.perf_counter() - start)
        raise
    finally:
        save()
    return output


def main():
    parser = argparse.ArgumentParser(description="Prepare inputs and run the configured Hinpost curriculum.")
    parser.add_argument("--config", default="configs/hinpost.json")
    parser.add_argument("--mode", choices=["prepare", "train", "all"], default="all")
    parser.add_argument("--pcap-root", help="Class folders with one bidirectional flow per capture.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", help="New directory for per-seed training results.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--pretrained")
    source.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", help="Override the configuration's training seeds.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu-threads", type=int, default=0)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.seeds:
            if len(set(args.seeds)) != len(args.seeds):
                raise ValueError("Training seeds must be unique.")
            config["training"]["seeds"] = args.seeds
        if args.mode != "train" and not args.pcap_root:
            parser.error("Preparation requires --pcap-root.")
        if args.mode != "prepare":
            if not args.output or not (args.pretrained or args.from_scratch):
                parser.error("Training requires --output and either --pretrained or --from-scratch.")
            if Path(args.output).exists():
                parser.error("Choose a new output directory.")
            if args.pretrained and not Path(args.pretrained).is_file():
                parser.error("The initial checkpoint does not exist.")
        data_root = Path(args.data_root).resolve()
        if args.mode != "train":
            prepare_stages(config, args.pcap_root, data_root)
        if args.mode != "prepare":
            validate_prepared(data_root, config)
            output = Path(args.output).resolve()
            output.mkdir(parents=True, exist_ok=False)
            for seed in config["training"]["seeds"]:
                train_seed(config, data_root, output / f"seed-{seed}", seed, args.device, args.cpu_threads,
                           Path(args.pretrained).resolve() if args.pretrained else None)
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
