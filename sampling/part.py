"""part_id loading and validation, shared by every FALCON sampling stage.

Extracted from Ginex_with_intra/lib/epoch_sampler_intra.py so the node subsampler
and the intra-edge precompute read the partition through one code path.

part_id[v] is the cluster id of node v. It was produced over the SYMMETRIC CSC the
clustering ran on -- see cluster/*/conf.json ("csc_layout": "symmetric"). Because
part_id is a per-node property, an edge (u, v) is intra-cluster iff (v, u) is, so
symmetrizing the graph and intra-filtering it commute; the intra-edge precompute must
therefore run on that same symmetric edge set (Step C).
"""
from dataclasses import dataclass
import time

import torch


@dataclass
class PartInfo:
    """The partition and its derived shape."""
    part_id: torch.Tensor   # int64[N], cluster id of each node
    num_nodes: int
    num_clusters: int


def load_part_id(part_id_path, num_nodes, verbose=True):
    """Load part_id.pth, coerce to int64/cpu/contiguous, and validate its length.

    num_nodes is the node count of the graph FALCON is training on; part_id must cover
    exactly that graph or the cluster ids are meaningless against its edges.
    """
    t0 = time.time()
    part_id = torch.load(part_id_path, weights_only=False)
    if part_id.numel() != num_nodes:
        raise ValueError(
            f'part_id length {part_id.numel()} != num_nodes {num_nodes}; '
            f'the partition must be over the same graph FALCON is training on.'
        )
    # int64 so the same tensor feeds the C++-free intra precompute and any indexing.
    part_id = part_id.to(torch.int64).cpu().contiguous()
    num_clusters = int(part_id.max().item()) + 1

    if verbose:
        sizes = torch.bincount(part_id, minlength=num_clusters)
        nonempty = sizes[sizes > 0]
        print(f'[part] part_id={part_id_path}', flush=True)
        print(f'[part] {nonempty.numel()} non-empty clusters over {num_nodes:,} nodes, '
              f'sizes min={int(nonempty.min()):,} median={int(nonempty.median()):,} '
              f'max={int(nonempty.max()):,} (load {time.time() - t0:.1f}s)', flush=True)

    return PartInfo(part_id=part_id, num_nodes=int(num_nodes), num_clusters=num_clusters)
