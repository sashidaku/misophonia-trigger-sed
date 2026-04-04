import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import torch.nn as nn
from typing import List, Literal, Optional, Tuple
from dataclasses import dataclass
import torch.nn.functional as F

@torch.jit.script
def esn_forward_jit(
    x: torch.Tensor,
    h: torch.Tensor,
    W_in: torch.Tensor,
    W: torch.Tensor,
    bias: torch.Tensor,
    input_scale: float,
    leaking_rate: float,
) -> torch.Tensor:
    B, T, D = x.shape
    outputs = []

    for t in range(T):
        u_t = x[:, t, :]
        u_t = u_t * input_scale
        
        pre_act = F.linear(u_t, W_in)
        pre_act = pre_act + F.linear(h, W)
        pre_act = pre_act + bias
        
        h_new = torch.tanh(pre_act)
        h = (1.0 - leaking_rate) * h + leaking_rate * h_new
        outputs.append(h)
        
    return torch.stack(outputs, dim=1)

@torch.no_grad()
def estimate_spectral_radius_power(W: torch.Tensor, n_iter: int = 50, eps: float = 1e-12) -> torch.Tensor:
    H = W.size(0)
    v = torch.randn(H, device=W.device, dtype=W.dtype)
    v = v / (v.norm() + eps)
    for _ in range(n_iter):
        v = W @ v
        v = v / (v.norm() + eps)
    wv = W @ v
    sr = (wv.norm() + eps)
    return sr

def make_cycle_jump_W(H, jump=7, w_cycle=1.0, w_jump=0.5, device="cuda", dtype=torch.float16):
    W = torch.zeros(H, H, device=device, dtype=dtype)
    for i in range(H):
        W[i, (i-1) % H] = w_cycle
    for i in range(H):
        W[i, (i-jump) % H] = w_jump
        W[i, (i+jump) % H] = w_jump
    return W

class ESN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        spectral_radius: float = 0.9,
        leaking_rate: float = 1.0,
        density: float = 0.8,
        input_scale: float = 1.0,
        bias: bool = True,
        activation=F.tanh,
        device=None,
        dtype=None,
        topology: str = "random",
        jump: int = 7,
        w_cycle: float = 1.0,
        w_jump: float = 0.5,
        sr_power_iters: int = 50,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.leaking_rate = leaking_rate
        self.activation = activation
        self.input_scale = input_scale

        self.topology = topology
        self.jump = jump
        self.w_cycle = w_cycle
        self.w_jump = w_jump
        self.sr_power_iters = sr_power_iters

        # 入力 → reservoir
        self.W_in = nn.Parameter(
            torch.empty(hidden_dim, input_dim, **factory_kwargs),
            requires_grad=False
        )
        # reservoir 内の再帰
        self.W = nn.Parameter(
            torch.empty(hidden_dim, hidden_dim, **factory_kwargs),
            requires_grad=False
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(hidden_dim, **factory_kwargs),
                requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

        self._reset_parameters(spectral_radius, density)

    def _reset_parameters(self, spectral_radius: float, density: float):
        nn.init.uniform_(self.W_in, -0.5, 0.5)

        if self.topology == "crj":
            W = make_cycle_jump_W(
                self.hidden_dim,
                jump=self.jump,
                w_cycle=self.w_cycle,
                w_jump=self.w_jump,
                device=self.W.device,
                dtype=self.W.dtype,
            )
            sr = estimate_spectral_radius_power(W, n_iter=self.sr_power_iters)
            W = W * (spectral_radius / (sr + 1e-12))
        else:
            W = torch.randn(self.hidden_dim, self.hidden_dim,
                            device=self.W.device, dtype=self.W.dtype)

            if density < 1.0:
                mask = (torch.rand_like(W) < density).float()
                W = W * mask

                sr = estimate_spectral_radius_power(W, n_iter=self.sr_power_iters)
                W = W * (spectral_radius / (sr + 1e-12))

        with torch.no_grad():
            self.W.copy_(W)

        if self.bias is not None:
            nn.init.uniform_(self.bias, -0.1, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        assert D == self.input_dim

        h = x.new_zeros(B, self.hidden_dim)

        if self.bias is not None:
            bias_val = self.bias
        else:
            bias_val = torch.zeros(self.hidden_dim, device=x.device, dtype=x.dtype)

        return esn_forward_jit(
            x, h, self.W_in, self.W, bias_val, 
            self.input_scale, self.leaking_rate
        )


class BiESN(nn.Module):
    def __init__(
        self,
        *,
        base_esn_ctor,
        merge: str = "concat",
        share_weights: bool = False,
    ):
        super().__init__()
        assert merge in ["concat", "sum", "mean"]
        self.merge = merge
        self.share_weights = share_weights

        self.esn_f = base_esn_ctor()

        if share_weights:
            self.esn_b = self.esn_f
        else:
            self.esn_b = base_esn_ctor()

        H = self.esn_f.hidden_dim
        self.hidden_dim = (2 * H) if merge == "concat" else H

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_f = self.esn_f(x)

        x_rev = torch.flip(x, dims=[1])
        h_b_rev = self.esn_b(x_rev)
        h_b = torch.flip(h_b_rev, dims=[1])

        if self.merge == "concat":
            return torch.cat([h_f, h_b], dim=-1)
        elif self.merge == "sum":
            return h_f + h_b 
        else:
            return 0.5 * (h_f + h_b)
        
@dataclass(frozen=True)
class ESN_Para:
    H_ESN: int
    SPECTRAL_RADIUS: float
    LEAKING_RATE: float
    DENSITY: float
    INPUT_SCALE: float

    TOPOLOGY: Literal["random", "crj"] = "random"
    JUMP: int = 7
    W_CYCLE: float = 1.0
    W_JUMP: float = 0.5
    SR_POWER_ITERS: int = 50

    BIDIRECTIONAL: bool = True
    BI_MERGE: Literal["concat", "sum", "mean"] = "concat"
    BI_SHARE_WEIGHTS: bool = False

    @classmethod
    def from_cfg(cls, cfg) -> "ESN_Para":
        return cls(
            H_ESN=cfg.H_ESN,
            SPECTRAL_RADIUS=cfg.SPECTRAL_RADIUS,
            LEAKING_RATE=cfg.LEAKING_RATE,
            DENSITY=cfg.DENSITY,
            INPUT_SCALE=cfg.INPUT_SCALE,

            # cfg 側に無ければデフォルト
            TOPOLOGY=getattr(cfg, "TOPOLOGY", "random"),
            JUMP=getattr(cfg, "JUMP", 7),
            W_CYCLE=getattr(cfg, "W_CYCLE", 1.0),
            W_JUMP=getattr(cfg, "W_JUMP", 0.5),
            SR_POWER_ITERS=getattr(cfg, "SR_POWER_ITERS", 50),

            BIDIRECTIONAL=getattr(cfg, "BIDIRECTIONAL_ESN", False),
            BI_MERGE=getattr(cfg, "BI_MERGE", "concat"),
            BI_SHARE_WEIGHTS=getattr(cfg, "BI_SHARE_WEIGHTS", False),
        )
    
class ESNReadout(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, h):
        logits = self.fc(h)
        return logits
    
class GroupedESNReadout(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, group_indices: List[List[int]]):
        super().__init__()
        self.in_dim = in_dim
        self.n_classes = n_classes

        assert len(group_indices) >= 1
        self.group_indices = [torch.tensor(g, dtype=torch.long) for g in group_indices]

        self.heads = nn.ModuleList([
            nn.Linear(in_dim, len(g)) for g in group_indices
        ])

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        N = H.size(0)
        out = H.new_zeros((N, self.n_classes))
        for head, idxs in zip(self.heads, self.group_indices):
            idxs = idxs.to(H.device)
            out[:, idxs] = head(H)
        return out

def solve_ridge_regression(H, Y, l2_reg=1e-4):
    calc_device = H.device if H.is_cuda else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    N, Hdim = H.shape
    H_bias = torch.cat([H, torch.ones(N, 1, device=H.device, dtype=H.dtype)], dim=1)


    N, Hdim = H.shape
    _, C = Y.shape

    try:
        Hb_gpu = H_bias.to(calc_device).float()
        Y_gpu  = Y.to(calc_device).float()
        
        HtH = Hb_gpu.T @ Hb_gpu
        HtY = Hb_gpu.T @ Y_gpu
        
        del Hb_gpu, Y_gpu
        torch.cuda.empty_cache()
    except RuntimeError:
        print("Warning: Ridge calculation fallback to CPU due to OOM")
        H_bias = H_bias.float()
        Y = Y.float()
        HtH = H_bias.T @ H_bias
        HtY = H_bias.T @ Y

    reg = l2_reg * torch.eye(HtH.size(0), device=HtH.device)
    reg[-1, -1] = 0.0
    
    A = HtH + reg
    B = HtY
    
    W_full = torch.linalg.solve(A, B)
    
    W = W_full[:-1, :]
    b = W_full[-1, :]
    return W, b

def solve_ridge_grouped(H: torch.Tensor, Y: torch.Tensor, group_indices: List[List[int]], l2_reg: float):
    device = H.device
    Hdim = H.size(1)
    C_full = Y.size(1)

    W_full = torch.zeros((Hdim, C_full), device=device)
    b_full = torch.zeros((C_full,), device=device)

    for g in group_indices:
        print("ggggg", g)
        y_g = Y[:, g]
        Wg, bg = solve_ridge_regression(H, y_g, l2_reg=l2_reg)
        W_full[:, g] = Wg
        b_full[g] = bg

    return W_full, b_full

@dataclass(frozen=True)
class ReadoutTrainPara:
    LEARNING_RATE: float
    NUM_EPOCHS: int
    L2: float = 1e-4
    BATCH_SIZE: int = 4096
    PATIENCE: int = 10
    @classmethod
    def from_cfg(cls, cfg) -> "ReadoutTrainPara":
        return cls(
            LEARNING_RATE=cfg.learning_rate,
            NUM_EPOCHS=cfg.num_epochs,
            L2=getattr(cfg, "L2_REG", 1e-4),
            BATCH_SIZE=getattr(cfg, "BATCH_SIZE", 4096),
            PATIENCE=getattr(cfg, "PATIENCE", 10),
        )
    
def make_esn(
    *,
    ESN_cls,
    input_dim: int,
    device,
    cfg,
    esn_para: Optional[ESN_Para] = None,
) -> Tuple[object, ESN_Para]:
    para_used = esn_para if esn_para is not None else ESN_Para.from_cfg(cfg)
    print(para_used)
    def _base_ctor():
        return ESN_cls(
            input_dim=input_dim,
            hidden_dim=para_used.H_ESN,
            spectral_radius=para_used.SPECTRAL_RADIUS,
            leaking_rate=para_used.LEAKING_RATE,
            input_scale=para_used.INPUT_SCALE,
            density=para_used.DENSITY,
            device=device,
            topology=para_used.TOPOLOGY,
            jump=para_used.JUMP,
            w_cycle=para_used.W_CYCLE,
            w_jump=para_used.W_JUMP,
            sr_power_iters=para_used.SR_POWER_ITERS,
        )

    if getattr(para_used, "BIDIRECTIONAL", False):
        esn = BiESN(
            base_esn_ctor=_base_ctor,
            merge=getattr(para_used, "BI_MERGE", esn_para.BI_MERGE),
            share_weights=getattr(para_used, "BI_SHARE_WEIGHTS", False),
        ).to(device)
    else:
        esn = _base_ctor().to(device)

    return esn, para_used


class LinearWBReadout(nn.Module):
    def __init__(self, W: torch.Tensor, b: torch.Tensor):
        super().__init__()
        self.register_buffer("W", W) 
        self.register_buffer("b", b)
    def forward(self, H: torch.Tensor) -> torch.Tensor:
        return H @ self.W + self.b