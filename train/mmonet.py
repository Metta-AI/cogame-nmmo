"""Torch mirror of the puffernet MMONet (vendor/upstream/nmmo3.c demo).

Weight-file layout (flat float32, sequential; every segment is 8-float
aligned by construction so get_weights_aligned's rounding is vacuous):

    conv1.weight  128x59x5x5   conv1.bias 128
    conv2.weight  128x128x3x3  conv2.bias 128
    embed.weight  128x32
    proj.weight   512x1817     proj.bias  512
    decoder.weight 27x512                     (bias-free)
    mingru.proj[l].weight 1536x512, l=0..3    (bias-free)

Total 4,430,976 floats == nmmo3_weights.bin byte-exactly.

Forward (single step, matches nmmo3.c forward()):
    one-hot map (59ch, factors [4,4,17,5,3,5,5,5,7,4]) -> conv1(s3) ->
    relu -> conv2 -> flat 256
    player bytes (47) -> embed 47x32 -> flat 1504, plus raw floats 47
    reward bytes (10) raw
    cat 1817 -> proj -> relu -> 512
    4x MinGRU (highway): per layer proj hidden->3H (bias-free);
      h_tilde = x>=0 ? x+0.5 : sigmoid(x); s' = s + sig(g)*(h_tilde-s);
      out = sig(hw)*s' + (1-sig(hw))*x
    decoder 512 -> 27 = 26 action logits + 1 value
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FACTORS = [4, 4, 17, 5, 3, 5, 5, 5, 7, 4]
OFFSETS = [0] + list(np.cumsum(FACTORS)[:-1].tolist())
OBS_SIZE = 11 * 15 * 10 + 47 + 10
HIDDEN = 512
N_LOGITS = 26
N_LAYERS = 4


class MMONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(59, 128, 5, stride=3)
        self.conv2 = nn.Conv2d(128, 128, 3, stride=1)
        self.embed = nn.Embedding(128, 32)
        self.proj = nn.Linear(1817, HIDDEN)
        self.decoder = nn.Linear(HIDDEN, N_LOGITS + 1, bias=False)
        self.gru_proj = nn.ModuleList(
            [nn.Linear(HIDDEN, 3 * HIDDEN, bias=False)
             for _ in range(N_LAYERS)])
        offsets = torch.tensor(OFFSETS).view(1, -1, 1, 1)
        self.register_buffer("offsets", offsets)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, 1707) uint8 -> (B, 512) pre-recurrent features."""
        B = obs.shape[0]
        ob_map = obs[:, :1650].view(B, 11, 15, 10)
        mh = torch.zeros(B, 59, 11, 15, dtype=torch.float32,
                         device=obs.device)
        codes = ob_map.long().permute(0, 3, 1, 2) + self.offsets
        mh.scatter_(1, codes, 1.0)
        x = F.relu(self.conv1(mh))
        x = self.conv2(x).flatten(1)
        ob_player = obs[:, 1650:1697].long()
        pe = self.embed(ob_player).flatten(1)
        cat = torch.cat([x, pe, ob_player.float(),
                         obs[:, 1697:].float()], dim=1)
        return F.relu(self.proj(cat))

    def gru_step(self, x: torch.Tensor, state: torch.Tensor):
        """x: (B, H); state: (L, B, H). Returns (out, new_state)."""
        new_state = []
        for layer in range(N_LAYERS):
            c = self.gru_proj[layer](x)
            hidden, gate, hw = c.split(HIDDEN, dim=-1)
            s = state[layer]
            h_tilde = torch.where(hidden >= 0, hidden + 0.5,
                                  torch.sigmoid(hidden))
            s_new = s + torch.sigmoid(gate) * (h_tilde - s)
            hw_s = torch.sigmoid(hw)
            x = hw_s * s_new + (1.0 - hw_s) * x
            new_state.append(s_new)
        return x, torch.stack(new_state)

    def forward(self, obs: torch.Tensor, state: torch.Tensor):
        """Single step. obs (B, 1707) uint8; state (L, B, H).
        Returns (logits (B,26), value (B,), new_state)."""
        h = self.encode(obs)
        h, state = self.gru_step(h, state)
        out = self.decoder(h)
        return out[:, :N_LOGITS], out[:, N_LOGITS], state

    def initial_state(self, batch: int, device=None) -> torch.Tensor:
        return torch.zeros(N_LAYERS, batch, HIDDEN, device=device)


def _take(flat: np.ndarray, idx: int, shape) -> tuple[torch.Tensor, int]:
    n = int(np.prod(shape))
    t = torch.from_numpy(flat[idx:idx + n].copy()).view(*shape)
    return t, idx + n


def load_puffer_bin(path: str) -> MMONet:
    flat = np.fromfile(path, dtype=np.float32)
    net = MMONet()
    idx = 0
    sd = {}
    sd["conv1.weight"], idx = _take(flat, idx, (128, 59, 5, 5))
    sd["conv1.bias"], idx = _take(flat, idx, (128,))
    sd["conv2.weight"], idx = _take(flat, idx, (128, 128, 3, 3))
    sd["conv2.bias"], idx = _take(flat, idx, (128,))
    sd["embed.weight"], idx = _take(flat, idx, (128, 32))
    sd["proj.weight"], idx = _take(flat, idx, (512, 1817))
    sd["proj.bias"], idx = _take(flat, idx, (512,))
    sd["decoder.weight"], idx = _take(flat, idx, (27, 512))
    for layer in range(N_LAYERS):
        sd[f"gru_proj.{layer}.weight"], idx = _take(
            flat, idx, (3 * 512, 512))
    assert idx == flat.size, (idx, flat.size)
    net.load_state_dict(sd, strict=False)
    # keep the offsets buffer from __init__
    return net


def save_puffer_bin(net: MMONet, path: str) -> None:
    parts = [
        net.conv1.weight, net.conv1.bias,
        net.conv2.weight, net.conv2.bias,
        net.embed.weight,
        net.proj.weight, net.proj.bias,
        net.decoder.weight,
    ] + [net.gru_proj[layer].weight for layer in range(N_LAYERS)]
    flat = np.concatenate(
        [p.detach().cpu().numpy().astype(np.float32).ravel()
         for p in parts])
    assert flat.size == 4430976, flat.size
    flat.tofile(path)
