"""Validated hex-token inputs and paired-view batching."""
import csv
import torch
from uer.utils.constants import CLS_TOKEN, UNK_TOKEN

def read_dataset(args, path, is_training=False):
    """Read cleaned hex-token TSV data, with an independent mask for each view."""
    import re
    dataset = []
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not {"label", "text_a"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: expected label and text_a columns.")
        for line_number, row in enumerate(reader, 2):
            label = int(row["label"])
            if not 0 <= label < args.labels_num:
                raise ValueError(f"{path}:{line_number}: label is outside [0, labels_num).")
            views = row["text_a"].split("@@")
            paired = args.soft_targets and is_training
            if paired and len(views) != 2:
                raise ValueError(f"{path}:{line_number}: paired training requires exactly two views separated by @@.")
            if len(views) > 2:
                raise ValueError(f"{path}:{line_number}: multiple augmented views are unsupported.")
            encoded = []
            for text in views:
                words = text.split()
                if not words or any(not re.fullmatch(r"[0-9a-fA-F]{4}", w) for w in words):
                    raise ValueError(f"{path}:{line_number}: expected cleaned four-digit hex tokens; run prepare_data.py.")
                tokens = [CLS_TOKEN] + args.tokenizer.tokenize(text)
                if UNK_TOKEN in tokens:
                    raise ValueError(f"{path}:{line_number}: vocabulary cannot represent the input tokens.")
                ids = args.tokenizer.convert_tokens_to_ids(tokens)[:args.seq_length]
                mask = [1] * len(ids) + [0] * (args.seq_length - len(ids))
                ids += [0] * (args.seq_length - len(ids))
                encoded.append((ids, mask))
            dataset.append((encoded[0][0], label, encoded[0][1], encoded[1] if paired else None))
    if not dataset:
        raise ValueError(f"{path}: dataset is empty.")
    return dataset


def batches(dataset, batch_size, device):
    for start in range(0, len(dataset), batch_size):
        group = dataset[start:start + batch_size]
        src = torch.tensor([x[0] for x in group], dtype=torch.long, device=device)
        labels = torch.tensor([x[1] for x in group], dtype=torch.long, device=device)
        seg = torch.tensor([x[2] for x in group], dtype=torch.long, device=device)
        augmented = None
        if group[0][3] is not None:
            augmented = (torch.tensor([x[3][0] for x in group], dtype=torch.long, device=device),
                         torch.tensor([x[3][1] for x in group], dtype=torch.long, device=device))
        yield src, labels, seg, augmented


