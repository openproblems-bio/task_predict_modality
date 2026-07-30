"""BABEL-style spliced autoencoder (Wu et al., PNAS 2021, github.com/wukevin/babel).

Ported architecture: a dense Encoder/Decoder pair for RNA and a chromosome-split
ChromEncoder/ChromDecoder pair for ATAC, combined into an AssymSplicedAutoEncoder
that produces four translation paths (RNA->RNA, RNA->ATAC, ATAC->RNA, ATAC->ATAC)
from two separate (unshared) per-modality latent spaces.
"""

import torch
from torch import nn


class Exp(nn.Module):
    def forward(self, x):
        return torch.exp(torch.clamp(x, min=-15, max=15))


class ClippedSoftplus(nn.Module):
    def forward(self, x):
        return torch.clamp(nn.functional.softplus(x), min=1e-4, max=1e4)


def _init_linear(layer):
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def _as_list(activations, n):
    if activations is None:
        return [None] * n
    if isinstance(activations, nn.Module):
        return [activations] + [None] * (n - 1)
    activations = list(activations)
    return activations + [None] * (n - len(activations))


class Encoder(nn.Module):
    """Dense encoder: num_inputs -> 64 -> num_units."""

    def __init__(self, num_inputs, num_units=32, activation=nn.PReLU):
        super().__init__()
        self.net = nn.Sequential(
            _init_linear(nn.Linear(num_inputs, 64)),
            nn.BatchNorm1d(64),
            activation(),
            _init_linear(nn.Linear(64, num_units)),
            nn.BatchNorm1d(num_units),
            activation(),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """Dense decoder: num_units -> 64 -> three parallel num_outputs heads."""

    def __init__(self, num_units, num_outputs, final_activations=None, activation=nn.PReLU):
        super().__init__()
        self.shared = nn.Sequential(
            _init_linear(nn.Linear(num_units, 64)),
            nn.BatchNorm1d(64),
            activation(),
        )
        self.head1 = _init_linear(nn.Linear(64, num_outputs))
        self.head2 = _init_linear(nn.Linear(64, num_outputs))
        self.head3 = _init_linear(nn.Linear(64, num_outputs))
        acts = _as_list(final_activations, 3)
        self.act1, self.act2, self.act3 = acts

    def forward(self, x, size_factors=None):
        h = self.shared(x)
        r1 = self.head1(h)
        if self.act1 is not None:
            r1 = self.act1(r1)
        if size_factors is not None:
            r1 = r1 * size_factors
        r2 = self.head2(h)
        if self.act2 is not None:
            r2 = self.act2(r2)
        r3 = self.head3(h)
        if self.act3 is not None:
            r3 = self.act3(r3)
        return r1, r2, r3


class ChromEncoder(nn.Module):
    """Per-chromosome encoder: each chrom's features -> 32 -> 16, concat, -> latent_dim."""

    def __init__(self, num_inputs, latent_dim=32, activation=nn.PReLU):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                _init_linear(nn.Linear(n_i, 32)),
                nn.BatchNorm1d(32),
                activation(),
                _init_linear(nn.Linear(32, 16)),
                nn.BatchNorm1d(16),
                activation(),
            )
            for n_i in num_inputs
        ])
        self.chrom_sizes = list(num_inputs)
        self.combine = nn.Sequential(
            _init_linear(nn.Linear(16 * len(num_inputs), latent_dim)),
            nn.BatchNorm1d(latent_dim),
            activation(),
        )

    def forward(self, x_per_chrom):
        outs = [branch(x_i) for branch, x_i in zip(self.branches, x_per_chrom)]
        return self.combine(torch.cat(outs, dim=-1))


class ChromDecoder(nn.Module):
    """Inverse of ChromEncoder: latent_dim -> per-chrom chunks -> three parallel
    per-chrom output heads, concatenated back to genome-wide vectors."""

    def __init__(self, num_outputs, latent_dim=32, final_activations=None, activation=nn.PReLU):
        super().__init__()
        n_chroms = len(num_outputs)
        self.chrom_sizes = list(num_outputs)
        self.expand = nn.Sequential(
            _init_linear(nn.Linear(latent_dim, n_chroms * 16)),
            nn.BatchNorm1d(n_chroms * 16),
            activation(),
        )
        self.shared_branches = nn.ModuleList([
            nn.Sequential(
                _init_linear(nn.Linear(16, 32)),
                nn.BatchNorm1d(32),
                activation(),
            )
            for _ in num_outputs
        ])
        self.head1 = nn.ModuleList([_init_linear(nn.Linear(32, n_i)) for n_i in num_outputs])
        self.head2 = nn.ModuleList([_init_linear(nn.Linear(32, n_i)) for n_i in num_outputs])
        self.head3 = nn.ModuleList([_init_linear(nn.Linear(32, n_i)) for n_i in num_outputs])
        acts = _as_list(final_activations, 3)
        self.act1, self.act2, self.act3 = acts

    def forward(self, x):
        h = self.expand(x)
        chunks = torch.chunk(h, len(self.chrom_sizes), dim=-1)
        r1_parts, r2_parts, r3_parts = [], [], []
        for branch, h1, h2, h3, chunk in zip(
            self.shared_branches, self.head1, self.head2, self.head3, chunks
        ):
            hc = branch(chunk)
            r1_parts.append(h1(hc))
            r2_parts.append(h2(hc))
            r3_parts.append(h3(hc))
        r1 = torch.cat(r1_parts, dim=-1)
        r2 = torch.cat(r2_parts, dim=-1)
        r3 = torch.cat(r3_parts, dim=-1)
        if self.act1 is not None:
            r1 = self.act1(r1)
        if self.act2 is not None:
            r2 = self.act2(r2)
        if self.act3 is not None:
            r3 = self.act3(r3)
        return r1, r2, r3


class AssymSplicedAutoEncoder(nn.Module):
    """Domain 1 (RNA) uses dense Encoder/Decoder; domain 2 (ATAC) uses chromosome-split
    ChromEncoder/ChromDecoder. Four translation paths share the two encoders/decoders:
    RNA->RNA, RNA->ATAC, ATAC->RNA, ATAC->ATAC. Latent spaces are per-modality, not tied;
    alignment across modalities is enforced only via the training loss (QuadLoss), not
    architecturally.
    """

    def __init__(
        self,
        input_dim1,
        input_dim2,
        hidden_dim=16,
        final_activations1=None,
        final_activations2=None,
    ):
        super().__init__()
        final_activations1 = final_activations1 or [Exp(), ClippedSoftplus()]
        final_activations2 = final_activations2 or [nn.Sigmoid()]
        self.encoder1 = Encoder(input_dim1, num_units=hidden_dim)
        self.decoder1 = Decoder(hidden_dim, input_dim1, final_activations=final_activations1)
        self.encoder2 = ChromEncoder(input_dim2, latent_dim=hidden_dim)
        self.decoder2 = ChromDecoder(input_dim2, latent_dim=hidden_dim, final_activations=final_activations2)

    def encode1(self, x1):
        return self.encoder1(x1)

    def encode2(self, x2_per_chrom):
        return self.encoder2(x2_per_chrom)

    def forward(self, x1, x2_per_chrom, size_factors1=None):
        encoded1 = self.encoder1(x1)
        encoded2 = self.encoder2(x2_per_chrom)

        preds11 = self.decoder1(encoded1, size_factors=size_factors1)
        preds12 = self.decoder2(encoded1)
        preds21 = self.decoder1(encoded2, size_factors=size_factors1)
        preds22 = self.decoder2(encoded2)

        return preds11, preds12, preds21, preds22, encoded1, encoded2
