import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedcnnEncoder(nn.Module):
    """
    Gated CNN encoder.
    """
    def __init__(self, args):
        super(GatedcnnEncoder, self).__init__()
        self.layers_num = args.layers_num
        self.kernel_size = args.kernel_size
        self.block_size = args.block_size
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size

        # nn.Conv2d(input_channels, output_channels，kernel_size)
        # Input channels: 1; output channels: args.hidden_size.
        # kernel_size = (args.kernel_size, args.emb_size)
        self.conv_1 = nn.Conv2d(1, args.hidden_size, (args.kernel_size, args.emb_size))
        self.gate_1 = nn.Conv2d(1, args.hidden_size, (args.kernel_size, args.emb_size))

        self.conv_b1 = nn.Parameter(torch.randn(1, args.hidden_size, 1, 1))
        self.gate_b1 = nn.Parameter(torch.randn(1, args.hidden_size, 1, 1))

        self.conv = nn.ModuleList(
            [
                nn.Conv2d(args.hidden_size, args.hidden_size, (args.kernel_size, 1))
                for _ in range(args.layers_num - 1)
            ]
        )
        self.gate = nn.ModuleList(
            [
                nn.Conv2d(args.hidden_size, args.hidden_size, (args.kernel_size, 1))
                for _ in range(args.layers_num - 1)
            ]
        )

        self.conv_b = nn.ParameterList(
            nn.Parameter(torch.randn(1, args.hidden_size, 1, 1))
            for _ in range(args.layers_num - 1)
        )
        self.gate_b = nn.ParameterList(
            nn.Parameter(torch.randn(1, args.hidden_size, 1, 1))
            for _ in range(args.layers_num - 1)
        )

    def forward(self, emb, seg):
        batch_size, seq_length, _ = emb.size()

        padding = torch.zeros([batch_size, self.kernel_size-1, self.emb_size]).to(emb.device)
        emb = torch.cat([padding, emb], dim=1).unsqueeze(1)  # batch_size, 1, seq_length+width-1, emb_size

        hidden = self.conv_1(emb)
        hidden += self.conv_b1.repeat(1, 1, seq_length, 1)
        gate = self.gate_1(emb)
        gate += self.gate_b1.repeat(1, 1, seq_length, 1)
        hidden = hidden * torch.sigmoid(gate)

        res_input = hidden

        padding = torch.zeros([batch_size, self.hidden_size, self.kernel_size-1, 1]).to(emb.device)
        hidden = torch.cat([padding, hidden], dim=2)

        for i, (conv_i, gate_i) in enumerate(zip(self.conv, self.gate)):
            hidden, gate = conv_i(hidden), gate_i(hidden)
            hidden += self.conv_b[i].repeat(1, 1, seq_length, 1)
            gate += self.gate_b[i].repeat(1, 1, seq_length, 1)
            hidden = hidden * torch.sigmoid(gate)
            if (i + 1) % self.block_size == 0:
                hidden = hidden + res_input
                res_input = hidden
            hidden = torch.cat([padding, hidden], dim=2)

        hidden = hidden[:, :, self.kernel_size - 1:, :]
        output = hidden.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)

        return output


class PlaincnnEncoder(nn.Module):
    """
    CNN encoder without gating mechanism.
    """
    def __init__(self, args):
        super(PlaincnnEncoder, self).__init__()
        self.layers_num = args.layers_num
        self.kernel_size = args.kernel_size
        self.block_size = args.block_size
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size

        self.conv_1 = nn.Conv2d(1, args.hidden_size, (args.kernel_size, args.emb_size))
        self.conv_b1 = nn.Parameter(torch.randn(1, args.hidden_size, 1, 1))

        self.conv = nn.ModuleList(
            [
                nn.Conv2d(args.hidden_size, args.hidden_size, (args.kernel_size, 1))
                for _ in range(args.layers_num - 1)
            ]
        )
        self.conv_b = nn.ParameterList(
            nn.Parameter(torch.randn(1, args.hidden_size, 1, 1))
            for _ in range(args.layers_num - 1)
        )

    def forward(self, emb, seg):
        batch_size, seq_length, _ = emb.size()

        padding = torch.zeros([batch_size, self.kernel_size-1, self.emb_size]).to(emb.device)
        emb = torch.cat([padding, emb], dim=1).unsqueeze(1)  # [B, 1, L+K-1, D]

        hidden = self.conv_1(emb) + self.conv_b1.repeat(1, 1, seq_length, 1)
        hidden = F.relu(hidden)

        res_input = hidden

        padding = torch.zeros([batch_size, self.hidden_size, self.kernel_size-1, 1]).to(emb.device)
        hidden = torch.cat([padding, hidden], dim=2)

        for i, conv_i in enumerate(self.conv):
            hidden = conv_i(hidden) + self.conv_b[i].repeat(1, 1, seq_length, 1)
            hidden = F.relu(hidden)
            if (i + 1) % self.block_size == 0:
                hidden = hidden + res_input
                res_input = hidden
            hidden = torch.cat([padding, hidden], dim=2)

        hidden = hidden[:, :, self.kernel_size - 1:, :]
        output = hidden.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)

        return output


class DepthwisecnnEncoder(nn.Module):
    """
    Depthwise CNN encoder with optional residual connections.
    """
    def __init__(self, args):
        super(DepthwisecnnEncoder, self).__init__()
        self.layers_num = args.layers_num
        self.kernel_size = args.kernel_size
        self.block_size = args.block_size
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size

        # First projection to hidden size
        self.project = nn.Conv1d(self.emb_size, self.hidden_size, kernel_size=1)

        # Depthwise + Pointwise convolutions
        self.depthwise_layers = nn.ModuleList()
        self.pointwise_layers = nn.ModuleList()
        for _ in range(self.layers_num):
            self.depthwise_layers.append(
                nn.Conv1d(
                    self.hidden_size,
                    self.hidden_size,
                    kernel_size=self.kernel_size,
                    groups=self.hidden_size,  # Depthwise
                    padding=self.kernel_size - 1
                )
            )
            self.pointwise_layers.append(
                nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=1)  # Pointwise
            )

    def forward(self, emb, seg=None):
        # emb: [B, L, D] → transpose to [B, D, L]
        x = emb.transpose(1, 2)
        x = self.project(x)  # [B, hidden_size, L]
        
        res_input = x
        for i in range(self.layers_num):
            out = self.depthwise_layers[i](x)
            out = out[:, :, :x.shape[-1]]  # remove padding tail
            out = F.relu(out)
            out = self.pointwise_layers[i](out)
            out = F.relu(out)

            if (i + 1) % self.block_size == 0:
                x = x + out  # residual connection
                res_input = x
            else:
                x = out

        output = x.transpose(1, 2)  # [B, L, hidden_size]
        return output


class LightweightcnnEncoder(nn.Module):
    """
    Lightweight Convolutional Encoder.
    Implements softmax-normalized 1D depthwise convolution.
    """
    def __init__(self, args):
        super(LightweightcnnEncoder, self).__init__()
        self.layers_num = args.layers_num
        self.kernel_size = args.kernel_size
        self.block_size = args.block_size
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size

        # Project to hidden size
        self.project = nn.Conv1d(self.emb_size, self.hidden_size, kernel_size=1)

        # Lightweight convolution kernel weights (learned)
        self.weight = nn.ParameterList([
            nn.Parameter(torch.Tensor(self.hidden_size, 1, self.kernel_size))
            for _ in range(self.layers_num)
        ])

        self.bias = nn.ParameterList([
            nn.Parameter(torch.Tensor(self.hidden_size))
            for _ in range(self.layers_num)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        for w in self.weight:
            nn.init.xavier_uniform_(w)
        for b in self.bias:
            nn.init.constant_(b, 0.0)

    def forward(self, emb, seg=None):
        # emb: [B, L, D] -> [B, D, L]
        x = emb.transpose(1, 2)
        x = self.project(x)  # [B, hidden_size, L]

        res_input = x
        for i in range(self.layers_num):
            # softmax over kernel size for each channel
            weight = F.softmax(self.weight[i], dim=-1)  # [hidden_size, 1, K]
            bias = self.bias[i]

            # Apply depthwise convolution with shared kernel
            x_padded = F.pad(x, (self.kernel_size - 1, 0))  # left padding
            out = F.conv1d(x_padded, weight, bias=bias, groups=self.hidden_size)
            out = F.relu(out)

            if (i + 1) % self.block_size == 0:
                x = x + out  # residual
                res_input = x
            else:
                x = out

        output = x.transpose(1, 2)  # [B, L, hidden_size]
        return output


class DynamiccnnEncoder(nn.Module):
    """
    Dynamic Convolutional Encoder.
    Generates a dynamic kernel for each time step using input features.
    """
    def __init__(self, args):
        super(DynamiccnnEncoder, self).__init__()
        self.layers_num = args.layers_num
        self.kernel_size = args.kernel_size
        self.block_size = args.block_size
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size

        # Project the input to the hidden dimension.
        self.project = nn.Conv1d(self.emb_size, self.hidden_size, kernel_size=1)

        # Generate a kernel weight vector at each position.
        self.dynamic_kernels = nn.ModuleList([
            nn.Conv1d(self.hidden_size, self.hidden_size * self.kernel_size, kernel_size=1)
            for _ in range(self.layers_num)
        ])

        self.bias = nn.ParameterList([
            nn.Parameter(torch.zeros(self.hidden_size))
            for _ in range(self.layers_num)
        ])

    def forward(self, emb, seg=None):
        # emb: [B, L, D] -> [B, D, L]
        x = emb.transpose(1, 2)
        x = self.project(x)  # [B, hidden_size, L]
        B, C, L = x.shape

        res_input = x

        for i in range(self.layers_num):
            # Generate per-step kernel weights: [B, C*K, L]
            kernels = self.dynamic_kernels[i](x)
            kernels = kernels.view(B, C, self.kernel_size, L)  # [B, C, K, L]
            kernels = F.softmax(kernels, dim=2)  # Normalize over the kernel-size dimension.

            # Padded input: [B, C, L + K - 1]
            x_padded = F.pad(x, (self.kernel_size - 1, 0))  # Left-pad with zeros.

            # Build sliding windows: [B, C, K, L]
            x_windows = x_padded.unfold(dimension=2, size=self.kernel_size, step=1)  # shape: [B, C, K, L]
            x_windows = x_windows.permute(0, 1, 3, 2).contiguous()  # → [B, C, K, L]

            

            # Apply dynamic weighted convolution.
            out = (x_windows * kernels).sum(dim=2) + self.bias[i].view(1, -1, 1)  # [B, C, L]
            out = F.relu(out)

            # Residual connection.
            if (i + 1) % self.block_size == 0:
                x = x + out
                res_input = x
            else:
                x = out

        output = x.transpose(1, 2)  # [B, L, C]
        return output


# class GatedLightweightCNNEncoder(nn.Module):
#     def __init__(self, args):
#         super(GatedLightweightCNNEncoder, self).__init__()
#         self.layers_num = args.layers_num
#         self.kernel_size = args.kernel_size
#         self.block_size = args.block_size
#         self.emb_size = args.emb_size
#         self.hidden_size = args.hidden_size

#         # Project emb_size to hidden_size first.
#         self.project = nn.Conv1d(self.emb_size, self.hidden_size, kernel_size=1)

#         # Define lightweight kernels for each layer.
#         self.kernel_h = nn.ParameterList([
#             nn.Parameter(torch.Tensor(self.hidden_size, self.kernel_size))
#             for _ in range(self.layers_num)
#         ])

#         self.bias_h = nn.ParameterList([
#             nn.Parameter(torch.zeros(self.hidden_size))
#             for _ in range(self.layers_num)
#         ])

#         self.kernel_g = nn.ParameterList([
#             nn.Parameter(torch.Tensor(self.hidden_size, self.kernel_size))
#             for _ in range(self.layers_num)
#         ])

#         self.bias_g = nn.ParameterList([
#             nn.Parameter(torch.zeros(self.hidden_size))
#             for _ in range(self.layers_num)
#         ])

#         self.reset_parameters()

#     def reset_parameters(self):
#         for w in self.kernel_h:
#             nn.init.xavier_uniform_(w.unsqueeze(1))
#         for w in self.kernel_g:
#             nn.init.xavier_uniform_(w.unsqueeze(1))
#         for b in self.bias_h:
#             nn.init.constant_(b, 0.0)
#         for b in self.bias_g:
#             nn.init.constant_(b, 0.0)

#     def forward(self, emb, seg=None):
#         # emb: [B, L, D] -> [B, D, L]
#         x = emb.transpose(1, 2)        # [B, D, L]
#         x = self.project(x)            # [B, hidden_size, L]
#         B, C, L = x.shape

#         res_input = x

#         for i in range(self.layers_num):
#             # kernel_h: [C, K] -> softmax across K
#             weight_h = F.softmax(self.kernel_h[i], dim=-1).unsqueeze(0).unsqueeze(-1)  # [1, C, K, 1]
#             weight_g = F.softmax(self.kernel_g[i], dim=-1).unsqueeze(0).unsqueeze(-1)  # [1, C, K, 1]


#             # pad input
#             x_padded = F.pad(x, (self.kernel_size - 1, 0))  # [B, C, L + K - 1]

#             # unfold into windows: [B, C, L, K]
#             x_windows = x_padded.unfold(dimension=2, size=self.kernel_size, step=1)
#             # -> permute to [B, C, K, L]
#             x_windows = x_windows.permute(0, 1, 3, 2).contiguous()

#             # apply lightweight conv
#             h_out = (x_windows * weight_h).sum(dim=2) + self.bias_h[i].view(1, -1, 1)
#             g_out = (x_windows * weight_g).sum(dim=2) + self.bias_g[i].view(1, -1, 1)

#             # gating
#             out = h_out * torch.sigmoid(g_out)
#             out = F.relu(out)

#             # residual every block_size layers
#             if (i + 1) % self.block_size == 0:
#                 x = x + out
#                 res_input = x
#             else:
#                 x = out

#         output = x.transpose(1, 2)  # [B, L, hidden_size]
#         return output

class GatedLightweightCNNEncoder(nn.Module):
    """
    Gated Lightweight CNN Encoder - improved version.
    Combines lightweight depthwise convolution with gating mechanism.
    """
    def __init__(self, args):
        super(GatedLightweightCNNEncoder, self).__init__()
        self.layers_num = args.layers_num
        self.kernel_size = args.kernel_size
        self.block_size = args.block_size
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size

        self.project = nn.Conv1d(self.emb_size, self.hidden_size, kernel_size=1)

        # lightweight conv kernels for h and g branches
        self.weight_h = nn.ParameterList([
            nn.Parameter(torch.Tensor(self.hidden_size, 1, self.kernel_size))
            for _ in range(self.layers_num)
        ])
        self.bias_h = nn.ParameterList([
            nn.Parameter(torch.zeros(self.hidden_size))
            for _ in range(self.layers_num)
        ])

        self.weight_g = nn.ParameterList([
            nn.Parameter(torch.Tensor(self.hidden_size, 1, self.kernel_size))
            for _ in range(self.layers_num)
        ])
        self.bias_g = nn.ParameterList([
            nn.Parameter(torch.zeros(self.hidden_size))
            for _ in range(self.layers_num)
        ])

        # optional: layer norm for each block
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_size) 
            for _ in range((self.layers_num + self.block_size - 1) // self.block_size)
        ])

        self.dropout = nn.Dropout(args.dropout) if hasattr(args, 'dropout') else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        for w in self.weight_h:
            nn.init.xavier_uniform_(w)
        for w in self.weight_g:
            nn.init.xavier_uniform_(w)
        for b in self.bias_h:
            nn.init.constant_(b, 0.0)
        for b in self.bias_g:
            nn.init.constant_(b, 0.0)

    def forward(self, emb, seg=None):
        # emb: [B, L, D]
        x = emb.transpose(1, 2)         # [B, D, L]
        x = self.project(x)             # [B, hidden_size, L]

        res_input = x
        norm_idx = 0

        for i in range(self.layers_num):
            weight_h = F.softmax(self.weight_h[i], dim=-1)  # [hidden_size, 1, kernel_size]
            weight_g = F.softmax(self.weight_g[i], dim=-1)

            bias_h = self.bias_h[i]
            bias_g = self.bias_g[i]

            x_padded = F.pad(x, (self.kernel_size - 1, 0))

            # depthwise conv
            h_out = F.conv1d(x_padded, weight_h, bias=bias_h, groups=self.hidden_size)
            g_out = F.conv1d(x_padded, weight_g, bias=bias_g, groups=self.hidden_size)

            out = h_out * torch.sigmoid(g_out)
            out = F.gelu(out)   # GELU or ReLU

            if (i + 1) % self.block_size == 0:
                out = out + res_input
                # LayerNorm over [B, L, C]
                out = out.transpose(1, 2)         # [B, L, C]
                out = self.layer_norms[norm_idx](out)
                out = self.dropout(out)
                out = out.transpose(1, 2)         # [B, C, L]

                res_input = out
                norm_idx += 1

            x = out

        # In case the total layers_num isn't a multiple of block_size:
        if (self.layers_num % self.block_size) != 0:
            out = out + res_input
            out = out.transpose(1, 2)
            out = self.layer_norms[-1](out)
            out = self.dropout(out)
            out = out.transpose(1, 2)

        output = out.transpose(1, 2)    # [B, L, hidden_size]
        return output
    

class GatedLightweightCNNEncoder_0707(nn.Module):
    """
    Gated Lightweight CNN Encoder - improved version.
    Combines lightweight depthwise convolution with gating mechanism.
    """
    def __init__(self, args):
        super(GatedLightweightCNNEncoder_0707, self).__init__()
        self.layers_num = args.layers_num
        self.kernel_size = args.kernel_size
        self.block_size = args.block_size
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size

        self.project = nn.Conv1d(self.emb_size, self.hidden_size, kernel_size=1)

        # convolution weights before softmax
        self.weight_h_raw = nn.ParameterList([
            nn.Parameter(torch.Tensor(self.hidden_size, 1, self.kernel_size))
            for _ in range(self.layers_num)
        ])
        self.bias_h = nn.ParameterList([
            nn.Parameter(torch.zeros(self.hidden_size))
            for _ in range(self.layers_num)
        ])

        self.weight_g_raw = nn.ParameterList([
            nn.Parameter(torch.Tensor(self.hidden_size, 1, self.kernel_size))
            for _ in range(self.layers_num)
        ])
        self.bias_g = nn.ParameterList([
            nn.Parameter(torch.zeros(self.hidden_size))
            for _ in range(self.layers_num)
        ])

        # precomputed softmaxed weights for inference
        self.register_buffer("weight_h_softmaxed", None)
        self.register_buffer("weight_g_softmaxed", None)

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_size)
            for _ in range((self.layers_num + self.block_size - 1) // self.block_size)
        ])

        self.dropout = nn.Dropout(args.dropout) if hasattr(args, 'dropout') else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        for w in self.weight_h_raw:
            nn.init.xavier_uniform_(w)
        for w in self.weight_g_raw:
            nn.init.xavier_uniform_(w)
        for b in self.bias_h:
            nn.init.constant_(b, 0.0)
        for b in self.bias_g:
            nn.init.constant_(b, 0.0)

        # Clear softmax buffers
        self.weight_h_softmaxed = None
        self.weight_g_softmaxed = None

    @torch.no_grad()
    def update_softmax_weights(self):
        """
        Precompute softmax weights for fast inference.
        Should be called after switching to eval() mode.
        """
        weight_h_list = []
        weight_g_list = []
        for i in range(self.layers_num):
            w_h = F.softmax(self.weight_h_raw[i], dim=-1)
            w_g = F.softmax(self.weight_g_raw[i], dim=-1)
            weight_h_list.append(w_h)
            weight_g_list.append(w_g)

        # Stack into tensor:
        # shape = [layers_num, hidden_size, 1, kernel_size]
        self.weight_h_softmaxed = torch.stack(weight_h_list, dim=0)
        self.weight_g_softmaxed = torch.stack(weight_g_list, dim=0)

    def forward(self, emb, seg=None):
        """
        emb: [B, L, D]
        output: [B, L, hidden_size]
        """
        x = emb.transpose(1, 2)         # [B, D, L]
        x = self.project(x)             # [B, hidden_size, L]

        res_input = x
        norm_idx = 0

        for i in range(self.layers_num):
            if self.training or self.weight_h_softmaxed is None:
                weight_h = F.softmax(self.weight_h_raw[i], dim=-1)   # [hidden_size, 1, kernel_size]
                weight_g = F.softmax(self.weight_g_raw[i], dim=-1)
            else:
                weight_h = self.weight_h_softmaxed[i]
                weight_g = self.weight_g_softmaxed[i]

            bias_h = self.bias_h[i]
            bias_g = self.bias_g[i]

            # non-causal padding (symmetric)
            pad_total = self.kernel_size - 1
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            x_padded = F.pad(x, (pad_left, pad_right))

            # depthwise conv
            h_out = F.conv1d(x_padded, weight_h, bias=bias_h, groups=self.hidden_size)
            g_out = F.conv1d(x_padded, weight_g, bias=bias_g, groups=self.hidden_size)

            out = h_out * torch.sigmoid(g_out)

            if (i + 1) % self.block_size == 0:
                out = out + 0.5 * res_input
                out = out.transpose(1, 2)         # [B, L, hidden_size]
                out = self.layer_norms[norm_idx](out)
                out = self.dropout(out)
                out = out.transpose(1, 2)         # [B, hidden_size, L]

                res_input = out
                norm_idx += 1

            x = out

        # In case layers_num not multiple of block_size
        if (self.layers_num % self.block_size) != 0:
            out = out + 0.5 * res_input
            out = out.transpose(1, 2)
            out = self.layer_norms[-1](out)
            out = self.dropout(out)
            out = out.transpose(1, 2)

        output = out.transpose(1, 2)    # [B, L, hidden_size]
        return output
    

class GatedLightweightCNNEncoder_0709(nn.Module):
    """
    Gated Lightweight CNN Encoder - optimized version.
    Keeps depthwise conv, reduces unnecessary transpose.
    """

    def __init__(self, args):
        super(GatedLightweightCNNEncoder_0709, self).__init__()
        self.layers_num = args.layers_num
        self.kernel_size = args.kernel_size
        self.block_size = args.block_size
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size

        self.project = nn.Conv1d(self.emb_size, self.hidden_size, kernel_size=1)

        self.weight_h_raw = nn.ParameterList([
            nn.Parameter(torch.Tensor(self.hidden_size, 1, self.kernel_size))
            for _ in range(self.layers_num)
        ])
        self.bias_h = nn.ParameterList([
            nn.Parameter(torch.zeros(self.hidden_size))
            for _ in range(self.layers_num)
        ])

        self.weight_g_raw = nn.ParameterList([
            nn.Parameter(torch.Tensor(self.hidden_size, 1, self.kernel_size))
            for _ in range(self.layers_num)
        ])
        self.bias_g = nn.ParameterList([
            nn.Parameter(torch.zeros(self.hidden_size))
            for _ in range(self.layers_num)
        ])

        self.register_buffer("weight_h_softmaxed", None)
        self.register_buffer("weight_g_softmaxed", None)

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_size)
            for _ in range((self.layers_num + self.block_size - 1) // self.block_size)
        ])

        self.dropout = nn.Dropout(args.dropout) if hasattr(args, 'dropout') else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        for w in self.weight_h_raw:
            nn.init.xavier_uniform_(w)
        for w in self.weight_g_raw:
            nn.init.xavier_uniform_(w)
        for b in self.bias_h:
            nn.init.constant_(b, 0.0)
        for b in self.bias_g:
            nn.init.constant_(b, 0.0)

        self.weight_h_softmaxed = None
        self.weight_g_softmaxed = None

    @torch.no_grad()
    def update_softmax_weights(self):
        weight_h_list = []
        weight_g_list = []
        for i in range(self.layers_num):
            w_h = F.softmax(self.weight_h_raw[i], dim=-1)
            w_g = F.softmax(self.weight_g_raw[i], dim=-1)
            weight_h_list.append(w_h)
            weight_g_list.append(w_g)

        self.weight_h_softmaxed = torch.stack(weight_h_list, dim=0)
        self.weight_g_softmaxed = torch.stack(weight_g_list, dim=0)

    def forward(self, emb, seg=None):
        """
        emb: [B, L, D]
        output: [B, L, hidden_size]
        """

        # ---------------------------------------
        # Initial projection
        # ---------------------------------------
        x = emb.transpose(1, 2)              # [B, D, L]
        x = self.project(x)                  # [B, hidden_size, L]

        # from here on, switch to [B, L, H] layout
        x = x.transpose(1, 2)                # [B, L, H]
        res_input = x
        norm_idx = 0

        for i in range(self.layers_num):

            if self.training or self.weight_h_softmaxed is None:
                weight_h = F.softmax(self.weight_h_raw[i], dim=-1)
                weight_g = F.softmax(self.weight_g_raw[i], dim=-1)
            else:
                weight_h = self.weight_h_softmaxed[i]
                weight_g = self.weight_g_softmaxed[i]

            bias_h = self.bias_h[i]
            bias_g = self.bias_g[i]

            # depthwise conv expects [B, H, L]
            x_conv = x.transpose(1, 2)           # [B, H, L]

            # symmetric padding
            pad_total = self.kernel_size - 1
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            x_padded = F.pad(x_conv, (pad_left, pad_right))

            h_out = F.conv1d(
                x_padded,
                weight_h,
                bias=bias_h,
                groups=self.hidden_size
            )       # [B, H, L]

            g_out = F.conv1d(
                x_padded,
                weight_g,
                bias=bias_g,
                groups=self.hidden_size
            )

            out_conv = h_out * torch.sigmoid(g_out)

            # back to [B, L, H]
            out = out_conv.transpose(1, 2)       # [B, L, H]

            if (i + 1) % self.block_size == 0:
                # residual + norm in [B, L, H]
                out = out + 0.5 * res_input
                out = self.layer_norms[norm_idx](out)
                out = self.dropout(out)

                res_input = out
                norm_idx += 1

            x = out

        if (self.layers_num % self.block_size) != 0:
            out = out + 0.5 * res_input
            out = self.layer_norms[-1](out)
            out = self.dropout(out)

        return out      # [B, L, hidden_size]
    


class GatedLightweightCNNEncoder_0707_NoSoftmax(nn.Module):
    """
    Gated Lightweight CNN Encoder without Softmax on convolution weights.
    This version is for ablation experiments.
    """
    def __init__(self, args):
        super(GatedLightweightCNNEncoder_0707_NoSoftmax, self).__init__()
        self.layers_num = args.layers_num
        self.kernel_size = args.kernel_size
        self.block_size = args.block_size
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size

        self.project = nn.Conv1d(self.emb_size, self.hidden_size, kernel_size=1)

        # convolution weights (NO softmax)
        self.weight_h_raw = nn.ParameterList([
            nn.Parameter(torch.Tensor(self.hidden_size, 1, self.kernel_size))
            for _ in range(self.layers_num)
        ])
        self.bias_h = nn.ParameterList([
            nn.Parameter(torch.zeros(self.hidden_size))
            for _ in range(self.layers_num)
        ])

        self.weight_g_raw = nn.ParameterList([
            nn.Parameter(torch.Tensor(self.hidden_size, 1, self.kernel_size))
            for _ in range(self.layers_num)
        ])
        self.bias_g = nn.ParameterList([
            nn.Parameter(torch.zeros(self.hidden_size))
            for _ in range(self.layers_num)
        ])

        # buffers for precomputed weights (still raw in this ablation)
        self.register_buffer("weight_h_softmaxed", None)
        self.register_buffer("weight_g_softmaxed", None)

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_size)
            for _ in range((self.layers_num + self.block_size - 1) // self.block_size)
        ])

        self.dropout = nn.Dropout(args.dropout) if hasattr(args, 'dropout') else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        for w in self.weight_h_raw:
            nn.init.xavier_uniform_(w)
        for w in self.weight_g_raw:
            nn.init.xavier_uniform_(w)
        for b in self.bias_h:
            nn.init.constant_(b, 0.0)
        for b in self.bias_g:
            nn.init.constant_(b, 0.0)

        # Clear softmax buffers
        self.weight_h_softmaxed = None
        self.weight_g_softmaxed = None

    @torch.no_grad()
    def update_softmax_weights(self):
        """
        Precompute weights for inference (NO softmax applied in this version).
        """
        weight_h_list = []
        weight_g_list = []
        for i in range(self.layers_num):
            w_h = self.weight_h_raw[i]
            w_g = self.weight_g_raw[i]
            weight_h_list.append(w_h)
            weight_g_list.append(w_g)

        self.weight_h_softmaxed = torch.stack(weight_h_list, dim=0)
        self.weight_g_softmaxed = torch.stack(weight_g_list, dim=0)

    def forward(self, emb, seg=None):
        """
        emb: [B, L, D]
        output: [B, L, hidden_size]
        """
        x = emb.transpose(1, 2)         # [B, D, L]
        x = self.project(x)             # [B, hidden_size, L]

        res_input = x
        norm_idx = 0

        for i in range(self.layers_num):
            if self.training or self.weight_h_softmaxed is None:
                weight_h = self.weight_h_raw[i]
                weight_g = self.weight_g_raw[i]
            else:
                weight_h = self.weight_h_softmaxed[i]
                weight_g = self.weight_g_softmaxed[i]

            bias_h = self.bias_h[i]
            bias_g = self.bias_g[i]

            # non-causal symmetric padding
            pad_total = self.kernel_size - 1
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            x_padded = F.pad(x, (pad_left, pad_right))

            # depthwise convolution
            h_out = F.conv1d(x_padded, weight_h, bias=bias_h, groups=self.hidden_size)
            g_out = F.conv1d(x_padded, weight_g, bias=bias_g, groups=self.hidden_size)

            out = h_out * torch.sigmoid(g_out)

            if (i + 1) % self.block_size == 0:
                out = out + 0.5 * res_input
                out = out.transpose(1, 2)         # [B, L, hidden_size]
                out = self.layer_norms[norm_idx](out)
                out = self.dropout(out)
                out = out.transpose(1, 2)         # [B, hidden_size, L]

                res_input = out
                norm_idx += 1

            x = out

        # In case layers_num not multiple of block_size
        if (self.layers_num % self.block_size) != 0:
            out = out + 0.5 * res_input
            out = out.transpose(1, 2)
            out = self.layer_norms[-1](out)
            out = self.dropout(out)
            out = out.transpose(1, 2)

        output = out.transpose(1, 2)    # [B, L, hidden_size]
        return output


class GatedLightweightCNNEncoder_0707_NoGate(nn.Module):
    """
    Gated Lightweight CNN Encoder
    -> Gating mechanism is REMOVED in this ablation version.
    """
    def __init__(self, args):
        super(GatedLightweightCNNEncoder_0707_NoGate, self).__init__()
        self.layers_num = args.layers_num
        self.kernel_size = args.kernel_size
        self.block_size = args.block_size
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size

        self.project = nn.Conv1d(self.emb_size, self.hidden_size, kernel_size=1)

        # convolution weights for h branch
        self.weight_h_raw = nn.ParameterList([
            nn.Parameter(torch.Tensor(self.hidden_size, 1, self.kernel_size))
            for _ in range(self.layers_num)
        ])
        self.bias_h = nn.ParameterList([
            nn.Parameter(torch.zeros(self.hidden_size))
            for _ in range(self.layers_num)
        ])

        # convolution weights for g branch (still defined but unused in forward)
        self.weight_g_raw = nn.ParameterList([
            nn.Parameter(torch.Tensor(self.hidden_size, 1, self.kernel_size))
            for _ in range(self.layers_num)
        ])
        self.bias_g = nn.ParameterList([
            nn.Parameter(torch.zeros(self.hidden_size))
            for _ in range(self.layers_num)
        ])

        # precomputed softmaxed weights for inference
        self.register_buffer("weight_h_softmaxed", None)
        self.register_buffer("weight_g_softmaxed", None)

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_size)
            for _ in range((self.layers_num + self.block_size - 1) // self.block_size)
        ])

        self.dropout = nn.Dropout(args.dropout) if hasattr(args, 'dropout') else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        for w in self.weight_h_raw:
            nn.init.xavier_uniform_(w)
        for w in self.weight_g_raw:
            nn.init.xavier_uniform_(w)
        for b in self.bias_h:
            nn.init.constant_(b, 0.0)
        for b in self.bias_g:
            nn.init.constant_(b, 0.0)

        self.weight_h_softmaxed = None
        self.weight_g_softmaxed = None

    @torch.no_grad()
    def update_softmax_weights(self):
        """
        Precompute softmax weights for fast inference.
        """
        weight_h_list = []
        weight_g_list = []
        for i in range(self.layers_num):
            w_h = F.softmax(self.weight_h_raw[i], dim=-1)
            w_g = F.softmax(self.weight_g_raw[i], dim=-1)
            weight_h_list.append(w_h)
            weight_g_list.append(w_g)

        self.weight_h_softmaxed = torch.stack(weight_h_list, dim=0)
        self.weight_g_softmaxed = torch.stack(weight_g_list, dim=0)

    def forward(self, emb, seg=None):
        """
        emb: [B, L, D]
        output: [B, L, hidden_size]
        """
        x = emb.transpose(1, 2)         # [B, D, L]
        x = self.project(x)             # [B, hidden_size, L]

        res_input = x
        norm_idx = 0

        for i in range(self.layers_num):
            if self.training or self.weight_h_softmaxed is None:
                weight_h = F.softmax(self.weight_h_raw[i], dim=-1)
                weight_g = F.softmax(self.weight_g_raw[i], dim=-1)
            else:
                weight_h = self.weight_h_softmaxed[i]
                weight_g = self.weight_g_softmaxed[i]

            bias_h = self.bias_h[i]
            bias_g = self.bias_g[i]

            pad_total = self.kernel_size - 1
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            x_padded = F.pad(x, (pad_left, pad_right))

            # depthwise convolution
            h_out = F.conv1d(x_padded, weight_h, bias=bias_h, groups=self.hidden_size)

            # g_out is computed but NOT used
            g_out = F.conv1d(x_padded, weight_g, bias=bias_g, groups=self.hidden_size)

            # Ablation: no gating
            out = h_out

            if (i + 1) % self.block_size == 0:
                out = out + 0.5 * res_input
                out = out.transpose(1, 2)         # [B, L, hidden_size]
                out = self.layer_norms[norm_idx](out)
                out = self.dropout(out)
                out = out.transpose(1, 2)         # [B, hidden_size, L]

                res_input = out
                norm_idx += 1

            x = out

        if (self.layers_num % self.block_size) != 0:
            out = out + 0.5 * res_input
            out = out.transpose(1, 2)
            out = self.layer_norms[-1](out)
            out = self.dropout(out)
            out = out.transpose(1, 2)

        output = out.transpose(1, 2)    # [B, L, hidden_size]
        return output


class GatedLightweightCNNEncoder_0707_ReLU(nn.Module):
    """
    Gated Lightweight CNN Encoder
    -> Gating mechanism is REMOVED in this ablation version.
    """
    def __init__(self, args):
        super(GatedLightweightCNNEncoder_0707_ReLU, self).__init__()
        self.layers_num = args.layers_num
        self.kernel_size = args.kernel_size
        self.block_size = args.block_size
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size

        self.project = nn.Conv1d(self.emb_size, self.hidden_size, kernel_size=1)

        # convolution weights for h branch
        self.weight_h_raw = nn.ParameterList([
            nn.Parameter(torch.Tensor(self.hidden_size, 1, self.kernel_size))
            for _ in range(self.layers_num)
        ])
        self.bias_h = nn.ParameterList([
            nn.Parameter(torch.zeros(self.hidden_size))
            for _ in range(self.layers_num)
        ])

        # convolution weights for g branch (still defined but unused in forward)
        self.weight_g_raw = nn.ParameterList([
            nn.Parameter(torch.Tensor(self.hidden_size, 1, self.kernel_size))
            for _ in range(self.layers_num)
        ])
        self.bias_g = nn.ParameterList([
            nn.Parameter(torch.zeros(self.hidden_size))
            for _ in range(self.layers_num)
        ])

        # precomputed softmaxed weights for inference
        self.register_buffer("weight_h_softmaxed", None)
        self.register_buffer("weight_g_softmaxed", None)

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_size)
            for _ in range((self.layers_num + self.block_size - 1) // self.block_size)
        ])

        self.dropout = nn.Dropout(args.dropout) if hasattr(args, 'dropout') else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        for w in self.weight_h_raw:
            nn.init.xavier_uniform_(w)
        for w in self.weight_g_raw:
            nn.init.xavier_uniform_(w)
        for b in self.bias_h:
            nn.init.constant_(b, 0.0)
        for b in self.bias_g:
            nn.init.constant_(b, 0.0)

        self.weight_h_softmaxed = None
        self.weight_g_softmaxed = None

    @torch.no_grad()
    def update_softmax_weights(self):
        """
        Precompute softmax weights for fast inference.
        """
        weight_h_list = []
        weight_g_list = []
        for i in range(self.layers_num):
            w_h = F.softmax(self.weight_h_raw[i], dim=-1)
            w_g = F.softmax(self.weight_g_raw[i], dim=-1)
            weight_h_list.append(w_h)
            weight_g_list.append(w_g)

        self.weight_h_softmaxed = torch.stack(weight_h_list, dim=0)
        self.weight_g_softmaxed = torch.stack(weight_g_list, dim=0)

    def forward(self, emb, seg=None):
        """
        emb: [B, L, D]
        output: [B, L, hidden_size]
        """
        x = emb.transpose(1, 2)         # [B, D, L]
        x = self.project(x)             # [B, hidden_size, L]

        res_input = x
        norm_idx = 0

        for i in range(self.layers_num):
            if self.training or self.weight_h_softmaxed is None:
                weight_h = F.softmax(self.weight_h_raw[i], dim=-1)
                weight_g = F.softmax(self.weight_g_raw[i], dim=-1)
            else:
                weight_h = self.weight_h_softmaxed[i]
                weight_g = self.weight_g_softmaxed[i]

            bias_h = self.bias_h[i]
            bias_g = self.bias_g[i]

            pad_total = self.kernel_size - 1
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            x_padded = F.pad(x, (pad_left, pad_right))

            # depthwise convolution
            h_out = F.conv1d(x_padded, weight_h, bias=bias_h, groups=self.hidden_size)

            # g_out is computed but NOT used
            g_out = F.conv1d(x_padded, weight_g, bias=bias_g, groups=self.hidden_size)

            # Ablation: no gating
            out = F.relu(h_out)

            if (i + 1) % self.block_size == 0:
                out = out + 0.5 * res_input
                out = out.transpose(1, 2)         # [B, L, hidden_size]
                out = self.layer_norms[norm_idx](out)
                out = self.dropout(out)
                out = out.transpose(1, 2)         # [B, hidden_size, L]

                res_input = out
                norm_idx += 1

            x = out

        if (self.layers_num % self.block_size) != 0:
            out = out + 0.5 * res_input
            out = out.transpose(1, 2)
            out = self.layer_norms[-1](out)
            out = self.dropout(out)
            out = out.transpose(1, 2)

        output = out.transpose(1, 2)    # [B, L, hidden_size]
        return output


class GatedLightweightCNNEncoder_0707_GeLU(nn.Module):
    """
    Gated Lightweight CNN Encoder
    -> Gating mechanism is REMOVED in this ablation version.
    """
    def __init__(self, args):
        super(GatedLightweightCNNEncoder_0707_GeLU, self).__init__()
        self.layers_num = args.layers_num
        self.kernel_size = args.kernel_size
        self.block_size = args.block_size
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size

        self.project = nn.Conv1d(self.emb_size, self.hidden_size, kernel_size=1)

        # convolution weights for h branch
        self.weight_h_raw = nn.ParameterList([
            nn.Parameter(torch.Tensor(self.hidden_size, 1, self.kernel_size))
            for _ in range(self.layers_num)
        ])
        self.bias_h = nn.ParameterList([
            nn.Parameter(torch.zeros(self.hidden_size))
            for _ in range(self.layers_num)
        ])

        # convolution weights for g branch (still defined but unused in forward)
        self.weight_g_raw = nn.ParameterList([
            nn.Parameter(torch.Tensor(self.hidden_size, 1, self.kernel_size))
            for _ in range(self.layers_num)
        ])
        self.bias_g = nn.ParameterList([
            nn.Parameter(torch.zeros(self.hidden_size))
            for _ in range(self.layers_num)
        ])

        # precomputed softmaxed weights for inference
        self.register_buffer("weight_h_softmaxed", None)
        self.register_buffer("weight_g_softmaxed", None)

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_size)
            for _ in range((self.layers_num + self.block_size - 1) // self.block_size)
        ])

        self.dropout = nn.Dropout(args.dropout) if hasattr(args, 'dropout') else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        for w in self.weight_h_raw:
            nn.init.xavier_uniform_(w)
        for w in self.weight_g_raw:
            nn.init.xavier_uniform_(w)
        for b in self.bias_h:
            nn.init.constant_(b, 0.0)
        for b in self.bias_g:
            nn.init.constant_(b, 0.0)

        self.weight_h_softmaxed = None
        self.weight_g_softmaxed = None

    @torch.no_grad()
    def update_softmax_weights(self):
        """
        Precompute softmax weights for fast inference.
        """
        weight_h_list = []
        weight_g_list = []
        for i in range(self.layers_num):
            w_h = F.softmax(self.weight_h_raw[i], dim=-1)
            w_g = F.softmax(self.weight_g_raw[i], dim=-1)
            weight_h_list.append(w_h)
            weight_g_list.append(w_g)

        self.weight_h_softmaxed = torch.stack(weight_h_list, dim=0)
        self.weight_g_softmaxed = torch.stack(weight_g_list, dim=0)

    def forward(self, emb, seg=None):
        """
        emb: [B, L, D]
        output: [B, L, hidden_size]
        """
        x = emb.transpose(1, 2)         # [B, D, L]
        x = self.project(x)             # [B, hidden_size, L]

        res_input = x
        norm_idx = 0

        for i in range(self.layers_num):
            if self.training or self.weight_h_softmaxed is None:
                weight_h = F.softmax(self.weight_h_raw[i], dim=-1)
                weight_g = F.softmax(self.weight_g_raw[i], dim=-1)
            else:
                weight_h = self.weight_h_softmaxed[i]
                weight_g = self.weight_g_softmaxed[i]

            bias_h = self.bias_h[i]
            bias_g = self.bias_g[i]

            pad_total = self.kernel_size - 1
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            x_padded = F.pad(x, (pad_left, pad_right))

            # depthwise convolution
            h_out = F.conv1d(x_padded, weight_h, bias=bias_h, groups=self.hidden_size)

            # g_out is computed but NOT used
            g_out = F.conv1d(x_padded, weight_g, bias=bias_g, groups=self.hidden_size)

            # Ablation: no gating
            out = F.gelu(h_out)

            if (i + 1) % self.block_size == 0:
                out = out + 0.5 * res_input
                out = out.transpose(1, 2)         # [B, L, hidden_size]
                out = self.layer_norms[norm_idx](out)
                out = self.dropout(out)
                out = out.transpose(1, 2)         # [B, hidden_size, L]

                res_input = out
                norm_idx += 1

            x = out

        if (self.layers_num % self.block_size) != 0:
            out = out + 0.5 * res_input
            out = out.transpose(1, 2)
            out = self.layer_norms[-1](out)
            out = self.dropout(out)
            out = out.transpose(1, 2)

        output = out.transpose(1, 2)    # [B, L, hidden_size]
        return output