"""Phase 1 per-epoch: materialize the RAM-resident small graph G*.  [Step D]

Consumes a Subsample (Step B) and the precomputed intra CSC (Step C) and produces the
in-memory graph the design's Phase 2 trains on: the subgraph induced on this epoch's
kept nodes, over intra-cluster edges only, with compact node ids laid out in cluster
order (so each cluster is a contiguous G* id block -- cluster_ptr from the Subsample).

Edges: an intra edge (u, v) survives iff BOTH endpoints are kept this epoch
(keep_mask[u] & keep_mask[v]); survivors are remapped u->new_id[u], v->new_id[v]. Since
the intra CSC is symmetric and the both-endpoints test is symmetric, G* is symmetric too
-- no reverse edge is added anywhere (the "대칭" was baked once in Step C).

Reading node u's intra row and filtering it to kept nodes yields exactly the induced-
subgraph row of new_id[u]; that is the whole construction. It is done blocked over kept
nodes so the pre-filter gather stays bounded on papers (the final G* is the deliverable
and lives in RAM: indices ~ p^2 * E_intra).

Features/labels are gathered from the mmap on kept_old; the gather is issued in ascending
old-id order for page locality, then scattered back into G* (cluster) order.
"""
from dataclasses import dataclass
from typing import Optional
import time

import numpy as np
import torch

from .node_subsampler import Subsample


@dataclass
class MaterializedGraph:
    """One epoch's RAM-resident G*. Node ids are compact (0..M-1), cluster-ordered."""
    indptr: torch.Tensor          # int64[M+1]
    indices: torch.Tensor         # int32[E_s], compact col ids, CSC (symmetric)
    cluster_ptr: torch.Tensor     # int64[C+1], G* id block boundary per cluster
    kept_old: torch.Tensor        # int64[M], compact id -> old id
    num_nodes: int                # M
    num_edges: int                # E_s
    x: Optional[torch.Tensor] = None           # [M, F]
    y: Optional[torch.Tensor] = None           # [M]
    train_mask: Optional[torch.Tensor] = None  # bool[M]
    val_mask: Optional[torch.Tensor] = None    # bool[M] (eval graph only)
    test_mask: Optional[torch.Tensor] = None   # bool[M] (eval graph only)
    prof: Optional[dict] = None                # stage seconds (see materialize's `prof`)


def materialize(sub: Subsample, intra_indptr, intra_indices,
                features=None, labels=None, train_mask_full=None,
                val_mask_full=None, test_mask_full=None,
                block_nodes=4_000_000, verbose=True) -> MaterializedGraph:
    """Build G* for one Subsample. intra_indptr/intra_indices are the Step C CSC
    (numpy / memmap). features/labels are full-graph mmaps indexed by OLD id; each is
    optional so the graph can be built without touching feature storage.

    val_mask_full/test_mask_full are the split indicators over the FULL node set (True on
    every val / test node -- nothing dropped). Pass them only when building the fixed
    evaluation graph (a full, un-subsampled G*); the per-epoch training graph needs only
    train_mask_full.
    """
    # prep이 epoch의 ~60%라 어느 단계가 먹는지 알아야 손을 댈 수 있다. 단계별 초를 모은다:
    #   edges  : intra 행 gather + 양끝점 필터 + 재번호 (intra_indices mmap 읽기)
    #   indptr : 차수 누적합
    #   feat   : feature gather (mmap -> 새 [M,F] 배열; papers f5 = 25.8GB)
    #   label  : 라벨/마스크 gather
    # 동작은 바뀌지 않는다 -- time.time() 호출만 추가.
    prof = {k: 0.0 for k in ('edges', 'indptr', 'feat', 'label')}
    t0 = time.time()
    keep_mask = sub.keep_mask.numpy()
    new_id = sub.new_id.numpy()
    kept_old = sub.kept_old.numpy()
    M = sub.num_kept
    iip = np.asarray(intra_indptr)
    t_stage = time.time()

    # ---- edges: blocked gather of each kept node's intra row, filter both endpoints ----
    gstar_deg = np.zeros(M, dtype=np.int64)
    col_blocks = []
    for j0 in range(0, M, block_nodes):
        j1 = min(j0 + block_nodes, M)
        rows_old = kept_old[j0:j1]                                  # old ids, G* order
        starts = iip[rows_old]
        seg_len = iip[rows_old + 1] - starts                        # intra-degree per node
        total = int(seg_len.sum())
        if total == 0:
            continue
        # index into intra_indices for every gathered edge: per row a contiguous run
        # [starts[k], starts[k]+seg_len[k]); concatenated in row order.
        row_local = np.repeat(np.arange(j1 - j0, dtype=np.int64), seg_len)
        run_base = np.repeat(np.cumsum(seg_len) - seg_len, seg_len)  # start-of-run per edge
        src_index = np.repeat(starts, seg_len) + (np.arange(total, dtype=np.int64) - run_base)
        cols_old = np.asarray(intra_indices[src_index]).astype(np.int64, copy=False)

        keep2 = keep_mask[cols_old]                                 # neighbor kept this epoch?
        cols_new = new_id[cols_old[keep2]].astype(np.int32, copy=False)
        gstar_deg[j0:j1] = np.bincount(row_local[keep2], minlength=(j1 - j0))
        col_blocks.append(cols_new)                                 # row-major, order preserved

    prof['edges'] = time.time() - t_stage; t_stage = time.time()

    gstar_indices = np.concatenate(col_blocks) if col_blocks else np.empty(0, np.int32)
    gstar_indptr = np.empty(M + 1, dtype=np.int64)
    gstar_indptr[0] = 0
    np.cumsum(gstar_deg, out=gstar_indptr[1:])
    E_s = int(gstar_indptr[-1])
    prof['indptr'] = time.time() - t_stage; t_stage = time.time()
    assert E_s == gstar_indices.shape[0], 'G* indptr total != indices length'

    g = MaterializedGraph(
        indptr=torch.from_numpy(gstar_indptr),
        indices=torch.from_numpy(gstar_indices),
        cluster_ptr=sub.cluster_ptr,
        kept_old=sub.kept_old,
        num_nodes=M,
        num_edges=E_s,
    )

    # ---- features / labels / train_mask: gather on kept_old ----
    if features is not None:
        Fdim = int(features.shape[1])
        x = np.empty((M, Fdim), dtype=features.dtype)
        order = np.argsort(kept_old)                                # ascending old id -> locality
        sorted_old = kept_old[order]
        # block-wise gather straight into x: caps the transient at one [BLK,F] block instead
        # of a full second [M,F] copy (papers f5: ~1GB vs ~26GB peak), same result as x[order]=...
        BLK = 2_000_000
        for b0 in range(0, M, BLK):
            b1 = min(b0 + BLK, M)
            x[order[b0:b1]] = features[sorted_old[b0:b1]]
        g.x = torch.from_numpy(x)
    prof['feat'] = time.time() - t_stage; t_stage = time.time()
    if labels is not None:
        g.y = torch.from_numpy(np.asarray(labels[kept_old]))
    if train_mask_full is not None:
        g.train_mask = torch.from_numpy(np.asarray(train_mask_full)[kept_old])
    if val_mask_full is not None:
        g.val_mask = torch.from_numpy(np.asarray(val_mask_full)[kept_old])
    if test_mask_full is not None:
        g.test_mask = torch.from_numpy(np.asarray(test_mask_full)[kept_old])

    prof['label'] = time.time() - t_stage
    g.prof = prof

    if verbose:
        ratio = E_s / iip[-1] if iip[-1] else 0.0
        feat = '' if g.x is None else f', x[{M:,},{g.x.shape[1]}]'
        total = time.time() - t0
        stages = '  '.join(f'{k} {prof[k]:.1f}s({prof[k] / max(total, 1e-9):.0%})'
                           for k in ('edges', 'indptr', 'feat', 'label'))
        print(f'[materialize] epoch {sub.epoch}: G* M={M:,} E_s={E_s:,} '
              f'(E_s/E_intra={ratio:.4f}){feat} ({total:.1f}s)\n'
              f'              stages: {stages}', flush=True)
    return g
