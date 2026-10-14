"""Per-epoch, cluster-preserving node subsampling for FALCON.  [Step B]

Ported from our Ginex-based prototype (ClusterNodeSampler). Same math:
keep a fraction p = 1/sqrt(factor) of the nodes uniformly at random *within each
cluster*, redrawn every epoch, so the induced subgraph has ~p^2 * E edges while every
cluster-size ratio and intra-/inter-cluster edge ratio is preserved (all scale by p^2).
Selection is stratified -- exactly round(p * size) nodes per cluster, never fewer than
1 -- so small clusters survive variance.

What FALCON adds over the GINEX version: GINEX handed a bool keep-mask to a C++ neighbor
sampler that walked the full disk graph. FALCON instead materializes a small in-memory
graph G*, so the subsample must also produce the compact relabeling G* needs:

    keep_mask[N]   bool     True iff node survives this epoch
    new_id[N]      int64    old id -> compact G* id, or -1 if dropped
    kept_old[M]    int64    compact id -> old id (feature/label gather uses this)
    cluster_ptr[]  int64    G* id boundaries of each non-empty cluster

kept_old is laid out in cluster order, so each cluster occupies a contiguous
[cluster_ptr[c], cluster_ptr[c+1]) block of G* ids -- cluster-batch slicing is O(1)
and no inter-cluster edges cross a block (design doc Phase 1-5).

This stage is symmetry-agnostic: it only reads part_id, never edges. The symmetric
edge set is handled once in intra_edges.py (Step C).
"""
from dataclasses import dataclass
import math
import time

import numpy as np
import torch

from .part import load_part_id


@dataclass
class Subsample:
    """One epoch's draw. All node-length tensors are indexed by OLD node id."""
    epoch: int
    keep_mask: torch.Tensor   # bool[N]
    new_id: torch.Tensor      # int64[N], old -> compact, -1 if dropped
    kept_old: torch.Tensor    # int64[M], compact -> old (cluster order)
    cluster_ptr: torch.Tensor # int64[num_clusters+1], G* id block boundaries
    num_kept: int             # M


class ClusterNodeSubsampler:
    """Draws a fresh stratified subsample over the full node set each epoch.

    The per-cluster node grouping is computed once in __init__; only the random keys
    and the per-cluster selection are redone per epoch, which is O(N) rather than the
    O(N log N) argsort every call.
    """

    def __init__(self, part_id_path, num_nodes, factor, seed=0, target_mask=None, verbose=True):
        if factor < 1:
            raise ValueError('factor must be >= 1 (factor=1 => p=1 => every node kept, '
                             'i.e. no subsampling).')

        info = load_part_id(part_id_path, num_nodes, verbose=verbose)
        part_id = info.part_id.numpy()

        self.num_nodes = info.num_nodes
        self.factor = float(factor)
        self.p = 1.0 / math.sqrt(factor)
        self.seed = int(seed)
        self.verbose = verbose

        # Group node ids by cluster once. order[starts[c]:starts[c]+sizes[c]] are the
        # old node ids of the c-th non-empty cluster; a random k-subset of that slice
        # is what survives. Empty cluster ids are collapsed away (np.unique keeps only
        # the ids that actually occur), so cluster_ptr never carries a zero-width block.
        t0 = time.time()
        self.order = np.argsort(part_id, kind='stable')
        self.cluster_ids, starts = np.unique(part_id[self.order], return_index=True)
        self.starts = starts.astype(np.int64)
        self.sizes = np.diff(np.append(self.starts, self.num_nodes))
        self.num_clusters = int(self.starts.shape[0])

        # Target (seed) nodes -- the nodes loss is computed on -- must NEVER be dropped by
        # the subsample: a target absent from G* is a target the model can't train on that
        # epoch. So each cluster keeps ALL its targets, plus a fraction p of its non-targets
        # (this is the GraphSAGE contract: seeds always in, only the neighborhood is sampled).
        # Without a target_mask every node is subsampled uniformly at rate p (old behavior).
        if target_mask is not None:
            self.target_ordered = np.asarray(target_mask).astype(bool)[self.order]
            n_tgt = np.add.reduceat(self.target_ordered.astype(np.int64), self.starts)
            n_nontgt = self.sizes - n_tgt
            self.keep_per_cluster = np.maximum(1, n_tgt + np.rint(self.p * n_nontgt).astype(np.int64))
        else:
            self.target_ordered = None
            self.keep_per_cluster = np.maximum(1, np.rint(self.p * self.sizes).astype(np.int64))
        del part_id

        if verbose:
            tgt_note = (f', all {int(self.target_ordered.sum()):,} targets force-kept'
                        if self.target_ordered is not None else '')
            print(f'[node-subsampler] factor={factor:g} -> p = 1/sqrt({factor:g}) = '
                  f'{self.p:.4f}, {self.num_clusters} non-empty clusters, expected kept '
                  f'nodes {int(self.keep_per_cluster.sum()):,}/{self.num_nodes:,}{tgt_note} '
                  f'(prep {time.time() - t0:.1f}s)', flush=True)

    def full(self):
        """A Subsample that keeps EVERY node, in cluster order (no subsampling).

        For the fixed evaluation graph: materialize() over this yields the full intra-
        only G* with all val/test nodes present. epoch=-1 marks it as not an epoch draw.
        """
        kept_old = self.order.astype(np.int64)          # all nodes, cluster order
        M = self.num_nodes
        new_id = np.empty(M, dtype=np.int64)
        new_id[kept_old] = np.arange(M, dtype=np.int64)
        keep_mask = np.ones(M, dtype=bool)
        cluster_ptr = np.append(self.starts, self.num_nodes).astype(np.int64)
        return Subsample(
            epoch=-1,
            keep_mask=torch.from_numpy(keep_mask),
            new_id=torch.from_numpy(new_id),
            kept_old=torch.from_numpy(kept_old),
            cluster_ptr=torch.from_numpy(cluster_ptr),
            num_kept=M,
        )

    def sample(self, epoch):
        """Draw epoch's subsample. Returns a Subsample (see class docstring)."""
        t0 = time.time()
        rng = np.random.default_rng(self.seed + epoch)
        # keys reordered into cluster-contiguous layout, so keys[start:start+size]
        # are exactly cluster c's keys.
        keys = rng.random(self.num_nodes, dtype=np.float64)[self.order]  # float64: avoid ~2^24 tie collisions

        # Force targets below every non-target key (keys are in [0, 1)). The per-cluster
        # k already budgets all targets (k >= n_targets), so the k lowest keys always
        # include every target -- they are guaranteed kept, only non-targets compete.
        if self.target_ordered is not None:
            keys[self.target_ordered] = -1.0

        kept_blocks = []
        cluster_ptr = np.zeros(self.num_clusters + 1, dtype=np.int64)
        for c in range(self.num_clusters):
            start = int(self.starts[c])
            size = int(self.sizes[c])
            k = int(self.keep_per_cluster[c])
            block = self.order[start:start + size]        # old ids in cluster c
            if k >= size:
                sel = block
            else:
                lowest = np.argpartition(keys[start:start + size], k)[:k]
                lowest.sort()                             # stable within-cluster order
                sel = block[lowest]
            kept_blocks.append(sel)
            cluster_ptr[c + 1] = cluster_ptr[c] + sel.shape[0]
        del keys

        kept_old = np.concatenate(kept_blocks) if kept_blocks else np.empty(0, np.int64)
        M = int(kept_old.shape[0])
        new_id = np.full(self.num_nodes, -1, dtype=np.int64)
        new_id[kept_old] = np.arange(M, dtype=np.int64)
        keep_mask = np.zeros(self.num_nodes, dtype=bool)
        keep_mask[kept_old] = True

        out = Subsample(
            epoch=int(epoch),
            keep_mask=torch.from_numpy(keep_mask),
            new_id=torch.from_numpy(new_id),
            kept_old=torch.from_numpy(kept_old),
            cluster_ptr=torch.from_numpy(cluster_ptr),
            num_kept=M,
        )
        if self.verbose:
            print(f'[node-subsampler] epoch {epoch}: kept {M:,}/{self.num_nodes:,} nodes '
                  f'({M / self.num_nodes:.4f}), expected edge ratio ~{1 / self.factor:.4f} '
                  f'({time.time() - t0:.1f}s)', flush=True)
        return out
