"""Package the supplied PCAP extractors with deterministic, reusable splits."""
import argparse
import csv
import hashlib
import importlib
import json
from pathlib import Path
import random
import re

METHODS = {
    "ether": "ether",
    "ip": "ip",
    "payload": "payload",
    "field-init": "field_init",
    "field-shuffle": "field_shuffle",
    "packet-shuffle": "packet_shuffle",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean_text(text):
    """Remove legacy packet delimiters/count metadata, preserving paired views."""
    views = []
    for view in text.split("@@"):
        view = re.sub(r"#\d+$", "", view)
        tokens = view.replace("@", " ").split()
        if not tokens or any(not re.fullmatch(r"[0-9a-fA-F]{4}", t) for t in tokens):
            raise ValueError("Expected nonempty sequences of four-digit hexadecimal tokens.")
        views.append(" ".join(tokens).lower())
    return "@@".join(views)


def validate_capture(path, method):
    """Reject input outside the network layers handled by each legacy extractor."""
    from scapy.all import IP, IPv6, TCP, UDP, rdpcap
    packets = rdpcap(str(path))
    if not packets:
        raise ValueError(f"Empty capture: {path.name}")
    keys = set()
    for packet in packets:
        network = packet[IP] if IP in packet else packet[IPv6] if IPv6 in packet else None
        if network is None or (TCP not in packet and UDP not in packet):
            raise ValueError(f"Expected IP TCP/UDP packets: {path.name}")
        if IPv6 in packet and method not in {"ether", "payload"}:
            raise ValueError(f"The {method} extractor does not support IPv6: {path.name}")
        transport = packet[TCP] if TCP in packet else packet[UDP]
        endpoints = sorted([(network.src, transport.sport), (network.dst, transport.dport)])
        keys.add((6 if TCP in packet else 17, tuple(endpoints)))
    if len(keys) != 1:
        raise ValueError(f"Expected one bidirectional five-tuple per file: {path.name}")
    times = [float(p.time) for p in packets]
    if times != sorted(times):
        raise ValueError(f"Packets must already be in chronological order: {path.name}")


def build_manifest(root, seed, samples_per_class):
    from sklearn.model_selection import StratifiedShuffleSplit
    categories = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if len(categories) < 2:
        raise ValueError("At least two class directories are required.")
    records, hashes = [], set()
    for label, category in enumerate(categories):
        candidates = sorted(p for p in (root / category).rglob("*")
                            if p.suffix.lower() in {".pcap", ".pcapng"} and p.is_file())
        rng = random.Random(seed + label)
        chosen = sorted(rng.sample(candidates, min(samples_per_class, len(candidates))))
        if not chosen:
            raise ValueError(f"No captures found for class {category}.")
        for p in chosen:
            digest = sha256(p)
            if digest in hashes:
                raise ValueError(f"Duplicate capture content: {p.relative_to(root)}")
            hashes.add(digest)
            records.append({"path": p.relative_to(root).as_posix(), "label": label, "sha256": digest})
    labels = [row["label"] for row in records]
    # Retain the supplied generator's 80/10/10 split and split seeds.
    first = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=41)
    train_ids, holdout_ids = next(first.split([[0]] * len(records), labels))
    second = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    test_ids, valid_ids = next(second.split([[0]] * len(holdout_ids), [labels[i] for i in holdout_ids]))
    for i in train_ids:
        records[int(i)]["split"] = "train"
    for i in test_ids:
        records[int(holdout_ids[i])]["split"] = "test"
    for i in valid_ids:
        records[int(holdout_ids[i])]["split"] = "valid"
    return {"schema_version": 1, "seed": seed, "split_seeds": [41, 42],
            "labels": {name: i for i, name in enumerate(categories)}, "records": records}


def prepare(args):
    root = Path(args.pcap_path).resolve()
    if not root.is_dir():
        raise ValueError("The PCAP root must be an existing directory.")
    destination = Path(args.dataset_dir)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Use an empty output directory to avoid mixing dataset versions.")
    if args.min_plen < 2 or args.aug_time != 1 or not 0 <= args.selected_rate <= 100:
        raise ValueError("Require min_plen >= 2, aug_time = 1, and selected_rate in [0, 100].")
    if args.samples_per_class < 1:
        raise ValueError("samples_per_class must be positive.")
    if args.split_manifest:
        manifest = json.loads(Path(args.split_manifest).read_text())
        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported split-manifest version.")
        if manifest.get("seed") != args.random_seed:
            raise ValueError("random_seed must match the reused manifest.")
    else:
        manifest = build_manifest(root, args.random_seed, args.samples_per_class)
    if args.class_num is not None and args.class_num != len(manifest["labels"]):
        raise ValueError("class_num differs from the manifest label count.")
    if not manifest["records"]:
        raise ValueError("The manifest contains no captures.")
    module = importlib.import_module("hinpost.data.operators." + METHODS[args.method])
    module.min_plen, module.aug_time = args.min_plen, args.aug_time
    module.selected_rate = args.selected_rate / 100
    rows = {name: [] for name in ("train", "valid", "test")}
    seen_paths, seen_hashes = set(), set()
    for record in manifest["records"]:
        path = (root / record["path"]).resolve()
        if root not in path.parents:
            raise ValueError("Manifest paths must stay inside the PCAP root.")
        if record["path"] in seen_paths or record["sha256"] in seen_hashes:
            raise ValueError("Duplicate path or capture hash in manifest.")
        seen_paths.add(record["path"])
        seen_hashes.add(record["sha256"])
        if record["label"] not in manifest["labels"].values() or record["split"] not in rows:
            raise ValueError("Invalid label or split in manifest.")
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Capture changed since manifest creation: {record['path']}")
        validate_capture(path, args.method)
        # Each flow has a stable seed, independent of extraction order and stage.
        flow_seed = int(hashlib.sha256(f"{args.random_seed}:{record['path']}".encode()).hexdigest()[:16], 16)
        random.seed(flow_seed)
        value = module.get_feature_packet(str(path), 128)
        if value == -1:
            raise ValueError(f"Extractor rejected {record['path']}; no split was silently changed.")
        text = clean_text(value[0])
        rows[record["split"]].append((record["label"], text))
    if any(not values for values in rows.values()):
        raise ValueError("Every split must be nonempty.")
    destination.mkdir(parents=True, exist_ok=True)
    for split, values in rows.items():
        with (destination / f"{split}_dataset.tsv").open("w", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(["label", "text_a"])
            writer.writerows(values)
    (destination / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    metadata = {"method": args.method, "random_seed": args.random_seed, "min_plen": args.min_plen,
                "aug_time": args.aug_time, "selected_rate": args.selected_rate,
                "counts": {k: len(v) for k, v in rows.items()},
                "address_policy": "legacy per-flow randomization; not field removal",
                "paper_template_search": False,
                "files": {p.name: sha256(p) for p in destination.glob("*_dataset.tsv")}}
    (destination / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--pcap_path", required=True, help="Class directories containing one flow per PCAP.")
    parser.add_argument("--dataset_dir", required=True, help="Empty output directory.")
    parser.add_argument("--class_num", type=int, help="Optional check against the discovered class count.")
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--samples_per_class", type=int, default=5000)
    parser.add_argument("--min_plen", type=int, default=32, help="Legacy minimum byte budget per packet.")
    parser.add_argument("--aug_time", type=int, default=1, help="Exactly one paired augmented view is supported.")
    parser.add_argument("--selected_rate", type=int, default=50, help="Percentage of fields to perturb.")
    parser.add_argument("--split_manifest", help="Reuse an existing flow selection and train/valid/test split.")
    args = parser.parse_args()
    try:
        prepare(args)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
