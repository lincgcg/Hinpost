"""Reload a completed run using its saved architecture and label count."""
import argparse
import json
from pathlib import Path
import sys
from hinpost.experiment import ROOT, execute
from hinpost.data.prepare import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="One completed seed directory.")
    parser.add_argument("--output", required=True, help="New directory for metrics, predictions and log.")
    parser.add_argument("--test-path", help="Override the test TSV location, e.g. after moving a run.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=0)
    args = parser.parse_args()
    run = Path(args.run).resolve()
    manifest = json.loads((run / "run.json").read_text())
    if manifest["status"] != "completed":
        parser.error("Evaluation requires a completed run.")
    config = manifest["configuration"]
    if sha256(run / "ip.bin") != manifest["checkpoint_sha256"]["ip"]:
        parser.error("Checkpoint differs from the completed run's recorded hash.")
    for asset in ("backbone", "vocabulary"):
        if sha256(ROOT / config[asset]) != manifest["asset_sha256"][asset]:
            parser.error(f"The {asset} differs from the completed run.")
    report = json.loads((run / "ip.json").read_text())
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-m", "hinpost.engine", "--evaluate_only",
               "--pretrained_model_path", str(run / "ip.bin"), "--config_path", str(ROOT / config["backbone"]),
               "--vocab_path", str(ROOT / config["vocabulary"]),
               "--test_path", args.test_path or report["configuration"]["test_path"],
               "--labels_num", str(len(json.loads((run / "labels.json").read_text()))),
               "--device", args.device, "--cpu_threads", str(args.cpu_threads),
               "--evaluate_output_path", str(output / "metrics.json"),
               "--predictions_path", str(output / "predictions.csv")]
    for key in ("expert_num", "top_k", "seq_length", "batch_size"):
        command += ["--" + key, str(config["training"][key])]
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    execute(command, output / "evaluation.log")
    print(f"Evaluation saved to {output}")


if __name__ == "__main__":
    main()
