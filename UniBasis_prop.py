UniBasis_prop.py的代码如下：

import math
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, get_laplacian
from utils import cheby

def presum_tensor(h, initial_val):
    length = len(h) + 1
    temp = torch.zeros(length, dtype=h.dtype, device=h.device)
    temp[0] = initial_val
    for idx in range(1, length):
        temp[idx] = temp[idx-1] + h[idx-1]
    return temp

def preminus_tensor(h, initial_val):
    length = len(h) + 1
    temp = torch.zeros(length, dtype=h.dtype, device=h.device)
    temp[0] = initial_val
    for idx in range(1, length):
        temp[idx] = temp[idx-1] - h[idx-1]
    return temp

@torch.jit.script_if_tracing
def _sparse_mv(edge_index, edge_weight, x):
    row, col = edge_index
    out = torch.zeros_like(x)
    out.index_add_(0, row, x[col] * edge_weight.view(-1, 1))
    return out

class UniBasisProp(MessagePassing):

    def __init__(self, K: int,
                 tau_mode: str = "blend",
                 tau_momentum: float = 0.9,
                 tau_alpha: float = 0.7,
                 tau_min: float = 0.2,
                 tau_max: float = 0.95,
                 **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.K = K


        self.initial_val_low  = Parameter(torch.tensor(2.0), requires_grad=False)
        self.initial_val_high = Parameter(torch.tensor(0.0), requires_grad=False)
        self.temp_low  = Parameter(torch.full((K,), 2.0 / K))
        self.temp_high = Parameter(torch.full((K,), 2.0 / K))


        self.register_buffer('tau_ma', torch.tensor(0.5))
        self.tau_mode = tau_mode
        self.tau_momentum = tau_momentum
        self.alpha = tau_alpha
        self.tau_min = tau_min
        self.tau_max = tau_max


        self.gamma_h = Parameter(torch.tensor(0.0))


        self.w_raw = Parameter(torch.zeros(K))

    def reset_parameters(self):
        with torch.no_grad():
            self.temp_low.fill_(2.0 / self.K)
            self.temp_high.fill_(2.0 / self.K)
            self.tau_ma.fill_(0.5)
            self.gamma_h.fill_(0.0)

    @torch.no_grad()
    def _estimate_tau(self, x: torch.Tensor, edge_index: torch.Tensor, update: bool) -> float:

        if not update:
            return float(self.tau_ma.item())

        edge_index_L, edge_weight_L = get_laplacian(edge_index, normalization='sym',
                                                    dtype=x.dtype, num_nodes=x.size(0))
        Lx   = _sparse_mv(edge_index_L, edge_weight_L, x)
        num  = (x * Lx).sum(dim=0)
        den  = (x * x).sum(dim=0) + 1e-12
        fdim = (num / (2.0 * den)).clamp(0.0, 1.0)
        f    = fdim.mean()

        row, col = edge_index
        x_row = F.normalize(x[row], p=2, dim=1)
        x_col = F.normalize(x[col], p=2, dim=1)
        cos = (x_row * x_col).sum(dim=1).clamp(-1.0, 1.0)
        tau_feat = 0.5 * (cos.mean() + 1.0)

        if self.tau_mode == "spectral":
            tau_curr = 1.0 - f
        elif self.tau_mode == "feature":
            tau_curr = tau_feat
        else:
            tau_curr = self.alpha * (1.0 - f) + (1.0 - self.alpha) * tau_feat


        self.tau_ma.mul_(self.tau_momentum).add_((1 - self.tau_momentum) * tau_curr)
        tau = float(self.tau_ma.item())
        tau = max(self.tau_min, min(self.tau_max, tau))
        return tau

    def forward(self, x, edge_index, edge_weight=None, highpass: bool = True, update_tau: bool = True):

        if highpass:
            TEMP = F.relu(self.temp_high)
            coe_tmp = presum_tensor(TEMP, self.initial_val_high)
        else:
            TEMP = F.relu(self.temp_low)
            coe_tmp = preminus_tensor(TEMP, self.initial_val_low)

        coe = coe_tmp.clone()
        K = self.K
        for i in range(K + 1):
            coe[i] = coe_tmp[0] * cheby(i, math.cos((K + 0.5) * math.pi / (K + 1)))
            for j in range(1, K + 1):
                x_j = math.cos((K - j + 0.5) * math.pi / (K + 1))
                coe[i] = coe[i] + coe_tmp[j] * cheby(i, x_j)
            coe[i] = 2 * coe[i] / (K + 1)


        edge_index1, norm1 = get_laplacian(edge_index, edge_weight, normalization='sym',
                                           dtype=x.dtype, num_nodes=x.size(self.node_dim))
        edge_index_tilde, norm_tilde = add_self_loops(edge_index1, norm1, fill_value=-1.0,
                                                      num_nodes=x.size(self.node_dim))


        tau = self._estimate_tau(x.detach(), edge_index, update=update_tau)
        tau_t = torch.tensor(tau, dtype=x.dtype, device=x.device)


        Tx_0 = x
        Tx_1 = self.propagate(edge_index_tilde, x=x, norm=norm_tilde, size=None)

        res_terms = [Tx_1 - Tx_0]

        out_cheb = coe[0] * 0.5 * Tx_0 + coe[1] * Tx_1
        last_1 = Tx_0
        last_2 = Tx_1
        for i in range(2, K + 1):
            Tx_2 = self.propagate(edge_index_tilde, x=last_2, norm=norm_tilde, size=None)
            Tx_2 = 2 * Tx_2 - last_1
            out_cheb = out_cheb + coe[i] * Tx_2


            res_terms.append(Tx_2 - last_1)

            last_1, last_2 = last_2, Tx_2


        w = F.softplus(self.w_raw)

        res_multi = 0.0
        for idx, r in enumerate(res_terms):

            if idx < w.shape[0]:
                res_multi = res_multi + w[idx] * r
            else:
                res_multi = res_multi + r


        w_sum = w.sum() + 1e-12
        res_multi = res_multi / w_sum

        out_cheb_norm = out_cheb.norm(p=2) + 1e-12
        res_norm = res_multi.norm(p=2) + 1e-12

        target_ratio = 0.1
        scale_factor = (out_cheb_norm / res_norm) * target_ratio
        scale_factor = torch.clamp(scale_factor, min=1e-3, max=10.0)
        res_multi = res_multi * scale_factor


        if highpass:
            out = out_cheb + self.gamma_h * (1.0 - tau_t) * res_multi
        else:

            low_inject_coef = 0.1
            out = out_cheb + low_inject_coef * self.gamma_h * tau_t * res_multi

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
