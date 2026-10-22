"""Top-K traffic experts and the supplied paired-view post-training objective.

Checkpoint parameter names follow the UER-format implementation.
"""
import torch
from torch import nn
from torch.nn import functional as F

from uer.embeddings import Embedding, str2embedding
from uer.encoders import str2encoder


class Expert(nn.Module):
    """Two-layer MLP mapping a flow representation to class logits."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, features):
        return self.net(features)


class MoE(nn.Module):
    """Select Top-K experts and sum their logits with normalized router weights."""

    def __init__(self, args):
        super().__init__()
        if not 1 <= args.top_k <= args.expert_num:
            raise ValueError("Require 1 <= top_k <= expert_num.")
        self.top_k = args.top_k
        self.labels_num = args.labels_num
        self.experts = nn.ModuleList(
            Expert(args.hidden_size, args.labels_num) for _ in range(args.expert_num)
        )
        self.router = nn.Linear(args.hidden_size, args.expert_num)

    def forward(self, features):
        scores = self.router(features)
        selected_scores, selected_ids = scores.topk(self.top_k, dim=1)
        sparse_scores = torch.full_like(scores, float("-inf"))
        sparse_scores = sparse_scores.scatter(-1, selected_ids, selected_scores)
        weights = F.softmax(sparse_scores, dim=-1)
        logits = features.new_zeros(features.size(0), self.labels_num)
        for index, expert in enumerate(self.experts):
            selected = (selected_ids == index).any(dim=-1)
            if selected.any():
                logits[selected] += expert(features[selected]) * weights[selected, index, None]
        return logits


class Hinpost(nn.Module):
    """Shared encoder with separate original/augmented expert classifiers.

    During paired training, original and augmented sequences are encoded in one
    batch with separate padding masks. Original-only inference uses raw_moe.
    The mse_output_layer names are historical checkpoint keys; their objective
    is normalized-projection L1, not logit MSE.
    """

    def __init__(self, args):
        super().__init__()
        self.embedding = Embedding(args)
        for name in args.embedding:
            self.embedding.update(str2embedding[name](args, len(args.tokenizer.vocab)), name)
        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha
        self.soft_beta = args.soft_beta
        if self.soft_targets:
            self.aug_moe = MoE(args)
            self.mse_output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
            self.mse_output_layer_2 = nn.Linear(args.hidden_size, 128)
        self.raw_moe = MoE(args)

    def project(self, representation):
        hidden = torch.tanh(self.mse_output_layer_1(representation))
        return F.normalize(self.mse_output_layer_2(hidden), dim=1)

    def forward(self, src, tgt, seg, soft_tgt=None):
        paired = self.soft_targets and soft_tgt is not None
        if paired:
            augmented_src, augmented_seg = soft_tgt
            tokens = torch.cat((src, augmented_src), dim=0)
            masks = torch.cat((seg, augmented_seg), dim=0)
            encoded = self.encoder(self.embedding(tokens, masks), masks)
            original, augmented = encoded[:, 0, :].chunk(2, dim=0)
            logits = self.raw_moe(original)
            augmented_logits = self.aug_moe(augmented)
        else:
            encoded = self.encoder(self.embedding(src, seg), seg)
            logits = self.raw_moe(encoded[:, 0, :])
        if tgt is None:
            return None, logits
        loss = F.nll_loss(F.log_softmax(logits, dim=-1), tgt.view(-1))
        if paired:
            # Retain the supplied three-term objective and parameterization.
            loss = (
                self.soft_beta * F.l1_loss(self.project(original), self.project(augmented))
                + self.soft_alpha * F.nll_loss(F.log_softmax(augmented_logits, dim=-1), tgt.view(-1))
                + loss
            )
        return loss, logits
