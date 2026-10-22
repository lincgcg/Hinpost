# Hinpost


Use Python 3.10 and install the dependencies before running the commands from the repository root:

```bash
pip install -r requirements.txt
```

## Contents

- [1. Protocol-Layered Input Construction](#1-protocol-layered-input-construction)
- [2. Category-Relation Expert Collaboration](#2-category-relation-expert-collaboration)
- [3. Hierarchical Curriculum Learning](#3-hierarchical-curriculum-learning)

## 1. Protocol-Layered Input Construction

The code is in [hinpost/data/](hinpost/data/). It converts flow captures into hexadecimal byte sequences and constructs the augmented inputs used in curriculum learning.

Arrange the dataset as `data/raw/<class_name>/<flow_name>.pcap`. Each capture should contain one bidirectional IPv4 TCP/UDP flow with packets in chronological order.

```bash
python run.py --config configs/hinpost.json --mode prepare \
              --pcap-root data/raw \
              --data-root data/prepared
```

- `pcap-root`: class directories containing per-flow captures.
- `data-root`: a new directory for the generated datasets.
- `data.seed` and `data.samples_per_class` in `configs/hinpost.json`: the sampling seed and maximum number of flows per class.

This produces `field-init`, `field-shuffle`, `packet-shuffle`, and `ip` datasets. Each contains `train_dataset.tsv`, `valid_dataset.tsv`, and `test_dataset.tsv`, using the same class mapping and 80/10/10 split. The first three contain original/augmented pairs; `ip` contains original inputs for the final stage.

The available input constructors use byte-length heuristics. Packet-attention and learned template search are not included.

## 2. Category-Relation Expert Collaboration

The code is in [hinpost/model.py](hinpost/model.py). `Expert` implements a two-layer MLP, and `MoE` implements the expert router and Top-K selection.

Set the following values in the `training` section of [configs/hinpost.json](configs/hinpost.json):

```json
{
  "expert_num": 8,
  "top_k": 4
}
```

- `expert_num`: the number of experts.
- `top_k`: the number of experts selected for each flow; it must not exceed `expert_num`.

The classifier output dimension is inferred from the dataset's class mapping. The current implementation combines selected expert logits by a gate-weighted sum. The concatenation/projection collaboration layer described in the manuscript is not included.

## 3. Hierarchical Curriculum Learning

The code is in [hinpost/experiment.py](hinpost/experiment.py) and [hinpost/engine.py](hinpost/engine.py). The curriculum trains the classifier in the following order:

| Stage | Input | Epochs |
|---|---|---:|
| 1 | Field initialization | 3 |
| 2 | Protocol shuffle | 3 |
| 3 | Packet shuffle | 3 |
| 4 | Original traffic | 1 |

Provide a compatible UER-format pretrained checkpoint and run:

```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
    --config configs/hinpost.json --mode train \
    --data-root data/prepared \
    --pretrained /path/to/pretrained_model.bin \
    --output outputs/hinpost \
    --seeds 1 --device cuda:0
```

- `pretrained`: a PyTorch state dictionary with `embedding.*` and `encoder.*` parameters matching the backbone configuration and vocabulary. Pretrained weights are not bundled.
- `output`: a new directory for trained checkpoints and results.
- `seeds`: training seeds; use `--seeds 1 2 3` for repeated runs.
- `training.batch_size`, `training.learning_rate`, and `training.seq_length`: training settings in `configs/hinpost.json`.
- `stages`: the epoch count and consistency weight for each stage.

Each stage starts from the previous stage's selected checkpoint. The first three stages use paired inputs with an auxiliary classification weight of 0.1 and normalized-representation L1 weights of 20, 10, and 20. The final stage uses original traffic only. Checkpoints are selected by validation accuracy, and the test set is evaluated after the final stage.
