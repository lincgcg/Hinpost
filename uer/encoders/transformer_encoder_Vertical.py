import torch
import torch.nn as nn
from uer.layers.transformer import TransformerLayer
from uer.layers.layer_norm import LayerNorm, T5LayerNorm
from uer.layers.relative_position_embedding import RelativePositionEmbedding

class TransformerEncoder_Vertical(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """
    def __init__(self, args):
        super(TransformerEncoder_Vertical, self).__init__()
        self.mask = args.mask
        self.layers_num = args.layers_num
        self.parameter_sharing = args.parameter_sharing
        self.factorized_embedding_parameterization = args.factorized_embedding_parameterization
        self.layernorm_positioning = args.layernorm_positioning
        self.relative_position_embedding = args.relative_position_embedding
        self.has_residual_attention = args.has_residual_attention

        has_bias = bool(1 - args.remove_transformer_bias)

        if self.factorized_embedding_parameterization:
            self.linear = nn.Linear(args.emb_size, args.hidden_size)

        if self.parameter_sharing:
            self.transformer = TransformerLayer(args)
        else:
            self.transformer = nn.ModuleList(
                [TransformerLayer(args) for _ in range(self.layers_num)]
            )

        if self.layernorm_positioning == "pre":
            if args.layernorm == "t5":
                self.layer_norm = T5LayerNorm(args.hidden_size)
            else:
                self.layer_norm = LayerNorm(args.hidden_size)

        if self.relative_position_embedding:
            self.relative_pos_emb = RelativePositionEmbedding(
                bidirectional=True,
                heads_num=args.heads_num,
                num_buckets=args.relative_attention_buckets_num
            )

        # ✅ Track internal states
        self.reset_state()

    def reset_state(self):
        """Reset to begin from the first layer."""
        self.current_layer = 0
        self.hidden = None
        self.prev_attn = None
        self.mask = None
        self.position_bias = None

    def prepare_inputs(self, emb, seg):
        """Prepare shared components before stepping through layers."""
        if self.factorized_embedding_parameterization:
            emb = self.linear(emb)

        batch_size, seq_length, _ = emb.size()

        # Prepare mask
        if self.mask == "fully_visible":
            mask = (seg > 0).unsqueeze(1).repeat(1, seq_length, 1).unsqueeze(1).float()
            mask = (1.0 - mask) * -10000.0
        elif self.mask == "causal":
            mask = torch.tril(torch.ones(seq_length, seq_length, device=emb.device))
            mask = (1.0 - mask) * -10000
            mask = mask.repeat(batch_size, 1, 1, 1)
        else:
            mask_a = (seg == 1).unsqueeze(1).repeat(1, seq_length, 1).unsqueeze(1).float()
            mask_b = (seg > 0).unsqueeze(1).repeat(1, seq_length, 1).unsqueeze(1).float()
            mask_tril = torch.tril(torch.ones(seq_length, seq_length, device=emb.device)).repeat(batch_size, 1, 1, 1)
            mask = (mask_a + mask_b + mask_tril >= 2).float()
            mask = (1.0 - mask) * -10000.0

        self.hidden = emb
        self.mask = mask

        if self.relative_position_embedding:
            self.position_bias = self.relative_pos_emb(emb, emb)

    def step(self):
        """Run a single transformer layer and update state."""
        if self.hidden is None or self.mask is None:
            raise RuntimeError("Must call prepare_inputs() before step().")

        if self.current_layer >= self.layers_num:
            raise IndexError("All transformer layers have already been used.")

        if self.parameter_sharing:
            self.hidden, self.prev_attn = self.transformer(
                self.hidden, self.mask, position_bias=self.position_bias,
                has_residual_attention=self.has_residual_attention,
                prev_attn=self.prev_attn
            )
        else:
            self.hidden, self.prev_attn = self.transformer[self.current_layer](
                self.hidden, self.mask, position_bias=self.position_bias,
                has_residual_attention=self.has_residual_attention,
                prev_attn=self.prev_attn
            )

        self.current_layer += 1

        if self.layernorm_positioning == "pre" and self.current_layer == self.layers_num:
            return self.layer_norm(self.hidden)
        return self.hidden
