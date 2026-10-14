"""FALCON 모델 — 미니 그래프에 적용하는 GCN / SAGE / GAT (PyG 기반).

세 모델 모두 `forward(x, edge_index) -> logits [num_nodes, num_classes]` 로 통일된
batch-agnostic 인터페이스를 가진다. "어떤 부분그래프를 먹이느냐"(클러스터 배치 vs
fanout 샘플)는 모델이 아니라 로더(sampling/masked_loader_*.py)가 결정한다.

  - GCN  : 정규화 인접행렬로 전체 이웃을 한 번에 (D^-1/2 A D^-1/2). 클러스터 배치에 사용.
  - SAGE : mean/max aggregator + self-concat, inductive. NeighborLoader의 fanout 서브그래프에 사용.
  - GAT  : 이웃마다 학습된 attention 가중치, multi-head. 클러스터 배치에 사용.

FALCON 전제: inter-cluster 엣지가 전처리에서 삭제됐으므로, 클러스터 배치는 경계 엣지가
없는 '완전한 부분그래프'다. 따라서 GCN 정규화·GAT attention이 누락 이웃 없이 정확하다.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


class GCN(nn.Module):
    """Kipf & Welling (2017). 전체 그래프(주어진 부분그래프) 정규화 인접행렬 전파."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 num_layers: int = 3, dropout: float = 0.5):
        super().__init__()
        assert num_layers >= 1
        self.dropout = dropout
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GCNConv(in_dim, out_dim))
        else:
            self.convs.append(GCNConv(in_dim, hidden))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden, hidden))
            self.convs.append(GCNConv(hidden, out_dim))

    def reset_parameters(self):
        for c in self.convs:
            c.reset_parameters()

    def forward(self, x, edge_index, cum_nodes=None, cum_edges=None):
        """Optional FALCON hop-prefix path, with the full sampled graph's GCN norm.

        No boundaries: the original GCNConv path (including its state_dict) is
        unchanged. With boundaries: return only seed rows, reusing one batch's
        normalization across layers. Dropout draws may differ from the full path.
        """
        if cum_nodes is None and cum_edges is None:
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
            return x
        if cum_nodes is None or cum_edges is None:
            raise ValueError('GCN prefix path requires both cum_nodes and cum_edges')
        from operator import index
        from torch_geometric.nn.conv.gcn_conv import gcn_norm

        try:
            cn = [index(v) for v in cum_nodes]
            ce = [index(v) for v in cum_edges]
        except TypeError as exc:
            raise ValueError('GCN hop boundaries must be integers') from exc
        layers = len(self.convs)
        if len(cn) != layers + 1 or len(ce) != layers:
            raise ValueError('GCN fanout hop count must equal the number of layers')
        if (edge_index.layout != torch.strided or edge_index.dtype != torch.long
                or edge_index.ndim != 2 or edge_index.size(0) != 2):
            raise ValueError('GCN prefix path requires int64 COO edge_index [2, E]')
        if (any(v < 0 for v in cn + ce) or cn != sorted(cn) or ce != sorted(ce)
                or cn[-1] != x.size(0) or ce[-1] != edge_index.size(1)):
            raise ValueError('Invalid GCN hop boundary sizes or ordering')
        # Each segment expands precisely the newly discovered previous frontier.
        # Check on device, with one scalar sync, before any truncation can hide
        # missing edges or invalid IDs. Repeated boundaries (empty hops) are valid.
        valid = torch.ones((), dtype=torch.bool, device=edge_index.device)
        edge_start = node_start = 0
        for hop, edge_end in enumerate(ce):
            src, dst = edge_index[:, edge_start:edge_end]
            valid = valid & ((src >= 0) & (src < cn[hop + 1])
                             & (dst >= node_start) & (dst < cn[hop])).all()
            edge_start, node_start = edge_end, cn[hop]
        if not bool(valid):
            raise ValueError('GCN edge order does not match FALCON hop prefixes')
        # The shared norm assumes the default unweighted GCN layers constructed
        # above. Reject custom configurations rather than silently changing math.
        if any(not c.normalize or not c.add_self_loops or c.improved or c.cached
               or c.flow != 'source_to_target' for c in self.convs):
            raise ValueError('GCN prefix path requires default uncached GCN normalization')

        ei, weights = gcn_norm(edge_index, num_nodes=x.size(0), dtype=x.dtype)
        # gcn_norm removes existing self-loops and appends one per node. Thus the
        # original cum_edges must be adjusted for removed loops before slicing.
        nonloop = edge_index[0] != edge_index[1]
        counts = torch.cat((nonloop.new_zeros(1, dtype=torch.long),
                            nonloop.to(torch.long).cumsum(0)))
        ends = counts[torch.tensor(ce, device=counts.device)].tolist()
        nonloop_end = ends[-1]
        loop_ei, loop_w = ei[:, nonloop_end:], weights[nonloop_end:]
        for k, conv in enumerate(self.convs, start=1):
            hop = layers - k
            n_out, edge_end = cn[hop], ends[hop]
            layer_ei = torch.cat((ei[:, :edge_end], loop_ei[:, :n_out]), dim=1)
            layer_w = torch.cat((weights[:edge_end], loop_w[:n_out]))
            # Rectangular adjacency: destination rows, source columns. Sparse mm
            # avoids materializing an E x hidden message tensor. Existing conv.lin
            # and conv.bias are reused, preserving checkpoint parameter names.
            adj = torch.sparse_coo_tensor(
                layer_ei.flip(0), layer_w, (n_out, x.size(0)),
                device=x.device, dtype=x.dtype).coalesce()
            x = torch.sparse.mm(adj, conv.lin(x))
            if conv.bias is not None:
                x = x + conv.bias
            if k < layers:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class SAGE(nn.Module):
    """GraphSAGE (Hamilton et al. 2017). inductive, aggregator + self-concat.

    fanout 이웃 샘플링은 NeighborLoader가 담당하므로(sampling 의 로더), 이 모듈은
    주어진 (샘플된) 서브그래프에서 aggregate만 한다. 레이어 수는 fanout 리스트 길이와
    맞춰야 한다(hop당 1레이어)."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 num_layers: int = 3, dropout: float = 0.5, aggr: str = "mean"):
        super().__init__()
        assert num_layers >= 1
        self.dropout = dropout
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(SAGEConv(in_dim, out_dim, aggr=aggr))
        else:
            self.convs.append(SAGEConv(in_dim, hidden, aggr=aggr))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden, hidden, aggr=aggr))
            self.convs.append(SAGEConv(hidden, out_dim, aggr=aggr))

    def reset_parameters(self):
        for c in self.convs:
            c.reset_parameters()

    def forward(self, x, edge_index, cum_nodes=None, cum_edges=None):
        """cum_* 가 주어지면 레이어별로 필요한 앞부분만 계산한다 (MFG 절단).

        병합 서브그래프를 그대로 L번 통과시키면 모든 레이어가 L홉 노드 전부를 계산하는데,
        정작 필요한 것은 마지막 레이어의 seed 행뿐이다. hop이 깊어질수록 노드 수가 기하적으로
        커지므로(papers 3층: seed 1,024 / hop1 7,964 / hop2 59,593 / hop3 331,055) 낭비가
        레이어마다 5.6배 -> 41.6배 -> 323배로 벌어진다. GEMM 총량 기준 18.7배.

        conv k(1-based)는 hop<=L-k 노드만 출력하면 되고, 그때 필요한 간선은 dst가 그 안에
        있는 것들뿐이다. 노드도 간선도 hop 순서로 쌓여 있으므로 둘 다 prefix 슬라이싱(뷰,
        복사 0)으로 얻어진다 -- DGL의 block처럼 id를 재매핑할 필요가 없다.

        cum_* 가 없으면(materialize 모드의 PyG NeighborLoader, 클러스터 배치 등) 기존 경로.
        결과는 두 경로가 동일하다 -- 절단은 쓰이지 않는 중간값을 안 만드는 것뿐이다.
        """
        if cum_nodes is None:
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
            return x

        L = len(self.convs)
        # hop 수와 레이어 수가 어긋나면 조용히 틀린 절단이 된다 -- 여기서 멈춘다.
        if len(cum_nodes) != L + 1 or len(cum_edges) != L:
            raise ValueError(f'fanout hop 수와 레이어 수가 다르다: num_layers={L} 인데 '
                             f'cum_nodes={len(cum_nodes)}(L+1이어야) '
                             f'cum_edges={len(cum_edges)}(L이어야)')
        for k, conv in enumerate(self.convs, start=1):
            n_out = cum_nodes[L - k]                       # 이 레이어가 내놓을 행 수
            ei = edge_index[:, :cum_edges[L - k]]          # dst가 그 안에 있는 간선만
            # (src, dst) 쌍: 메시지는 x 전체에서 뽑고, 결과는 앞 n_out행에 대해서만 낸다.
            x = conv((x, x[:n_out]), ei, size=(x.size(0), n_out))
            if k < L:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GAT(nn.Module):
    """Velickovic et al. (2018). multi-head attention. 중간 레이어는 head concat,
    마지막 레이어는 head 평균(concat=False)으로 out_dim을 맞춘다."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 num_layers: int = 3, heads: int = 8, dropout: float = 0.5):
        super().__init__()
        assert num_layers >= 1
        self.dropout = dropout
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GATConv(in_dim, out_dim, heads=1, concat=False, dropout=dropout))
        else:
            self.convs.append(GATConv(in_dim, hidden, heads=heads, dropout=dropout))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden * heads, hidden, heads=heads, dropout=dropout))
            self.convs.append(GATConv(hidden * heads, out_dim, heads=1, concat=False, dropout=dropout))

    def reset_parameters(self):
        for c in self.convs:
            c.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


def build_model(name: str, in_dim: int, hidden: int, out_dim: int,
                num_layers: int = 3, dropout: float = 0.5,
                heads: int = 8, aggr: str = "mean") -> nn.Module:
    name = name.lower()
    if name == "gcn":
        return GCN(in_dim, hidden, out_dim, num_layers, dropout)
    if name == "sage":
        return SAGE(in_dim, hidden, out_dim, num_layers, dropout, aggr=aggr)
    if name == "gat":
        return GAT(in_dim, hidden, out_dim, num_layers, heads=heads, dropout=dropout)
    raise ValueError(f"unknown model {name!r} (use gcn|sage|gat)")
