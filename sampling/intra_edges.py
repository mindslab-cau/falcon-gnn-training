"""Phase 0 precompute: the symmetric intra-cluster CSC.  [Step C]

Builds, once per (graph, partition) pair, the CSC of the graph with every
inter-cluster edge deleted -- each node's neighbor row keeps only neighbors in its
own cluster. Written next to part_id.pth so it is bound to that exact partition.

Symmetry: the INPUT CSC is already symmetric (ogbn-papers100M-sym has
csc_layout "symmetric"; ogbn-products is natively undirected). Intra-filtering
preserves that symmetry -- part_id is a per-node property, so (u,v) survives iff
(v,u) does -- so the reduced CSC is symmetric without adding any reverse edge. This
is where the design's "대칭 적용" happens, once, not per epoch.

Per epoch, materialize.py (Step D) reads this reduced CSC and, for each kept node,
filters its intra-neighbor row through keep_mask and remaps through new_id -- the same
row-filter GINEX did in C++, now in-memory over the already-inter-free graph.

Streaming: two passes over the edges in edge-balanced row blocks (mmap indices, never
fully resident). Pass 1 counts surviving neighbors per row -> intra_indptr; pass 2
writes the surviving neighbor ids -> intra_indices (int32; node ids < 2^31).

Sanity: the printed intra-edge ratio is also a correctness check on id alignment --
if part_id and the CSC disagreed on node ordering the ratio would collapse toward
1/num_clusters (random), not the high value a real partition gives.

CLI:  python -m sampling.intra_edges --dataset products
      python -m sampling.intra_edges --dataset papers   --part-k 1000
"""
import argparse
import json
import os
import time

import numpy as np

from .part import load_part_id


# Known (symmetric CSC dir, default part_id) pairs, relative to the repo root
# (the directory that holds sampling/). csc_dir holds indptr.dat / indices.dat /
# conf.json in the Ginex layout (see lib/data.py); part_id.pth is what
# preprocess/do_metis_new.py writes by default.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _dataset_paths(dataset, part_k):
    if dataset == 'products':
        csc = os.path.join(_ROOT, 'data/ogbn-products')
        part = os.path.join(_ROOT, f'cluster/ogbn-products-k{part_k}/metis/part_id.pth')
    elif dataset == 'papers':
        csc = os.path.join(_ROOT, 'data/ogbn-papers100M-sym')
        part = os.path.join(_ROOT, f'cluster/ogbn-papers100M-k{part_k}/metis/part_id.pth')
    elif dataset == 'friendster':
        csc = os.path.join(_ROOT, 'data/friendster/sym')
        part = os.path.join(_ROOT, f'cluster/friendster-k{part_k}/metis/part_id.pth')
    else:
        raise ValueError(f'unknown dataset {dataset!r} (use products|papers|friendster or pass explicit paths)')
    return csc, part


def _load_csc(csc_dir):
    """Return (indptr int64[N+1] in RAM, indices int64 memmap[E], conf dict)."""
    conf = json.load(open(os.path.join(csc_dir, 'conf.json')))
    indptr = np.fromfile(os.path.join(csc_dir, 'indptr.dat'),
                         dtype=conf['indptr_dtype']).reshape(tuple(conf['indptr_shape']))
    indices = np.memmap(os.path.join(csc_dir, 'indices.dat'), mode='r',
                        shape=tuple(conf['indices_shape']), dtype=conf['indices_dtype'])
    return indptr.astype(np.int64, copy=False), indices, conf


def _row_blocks(indptr, block_edges):
    """Yield (r0, r1) row ranges each spanning ~block_edges edges."""
    n = indptr.shape[0] - 1
    r0 = 0
    while r0 < n:
        target = indptr[r0] + block_edges
        r1 = int(np.searchsorted(indptr, target, side='left'))
        r1 = min(max(r1, r0 + 1), n)     # at least one row, never past the end
        yield r0, r1
        r0 = r1


def build_intra_csc(csc_dir, part_id_path, out_dir, block_edges=100_000_000, verbose=True):
    """Precompute and write the intra-cluster CSC. Returns a summary dict."""
    indptr, indices, conf = _load_csc(csc_dir)
    # materialize keeps G* symmetric ONLY if the source CSC is symmetric (the both-endpoints
    # filter is symmetric). Guard against silently pointing at a directed CSC (e.g. the
    # non-sym ogbn-papers100M): products has no csc_layout key (undirected native, ok).
    layout = conf.get('csc_layout')
    if layout not in (None, 'symmetric'):
        raise ValueError(
            f"intra_edges requires a symmetric source CSC, but {csc_dir} has "
            f"csc_layout={layout!r}; point --csc-dir at the symmetric CSC.")
    num_nodes = int(conf['num_nodes'])
    total_edges = int(indptr[-1])

    info = load_part_id(part_id_path, num_nodes, verbose=verbose)
    # int32 part id: num_clusters << 2^31, halves the resident/broadcast cost.
    part = info.part_id.numpy().astype(np.int32, copy=False)

    os.makedirs(out_dir, exist_ok=True)

    # ---- Single pass: read the CSC once, streaming surviving neighbors straight to
    # disk while tallying per-row counts. Kept edges leave each block in row-major /
    # within-row order, so appending them sequentially IS the final indices layout;
    # intra_indptr is just the cumsum of the counts gathered in the same pass. This
    # halves the I/O vs a count-then-write two-pass (papers reads 26GB once, not twice).
    t0 = time.time()
    intra_deg = np.zeros(num_nodes, dtype=np.int64)
    total_intra = 0
    out_indices_path = os.path.join(out_dir, 'indices.dat')
    with open(out_indices_path, 'wb') as out_indices:
        for r0, r1 in _row_blocks(indptr, block_edges):
            e0, e1 = int(indptr[r0]), int(indptr[r1])
            seg = np.asarray(indices[e0:e1])                   # neighbor ids, int64
            deg = (indptr[r0 + 1:r1 + 1] - indptr[r0:r1])      # per-row degree
            row_ids = np.repeat(np.arange(r1 - r0, dtype=np.int32), deg)
            keep = part[r0:r1][row_ids] == part[seg]           # src cluster == dst cluster
            intra_deg[r0:r1] = np.bincount(row_ids[keep], minlength=(r1 - r0))
            kept = np.ascontiguousarray(seg[keep], dtype=np.int32)  # order preserved
            out_indices.write(kept.tobytes())
            total_intra += kept.shape[0]

    intra_indptr = np.empty(num_nodes + 1, dtype=np.int64)
    intra_indptr[0] = 0
    np.cumsum(intra_deg, out=intra_indptr[1:])
    assert int(intra_indptr[-1]) == total_intra, 'indptr total drifted from bytes written'
    intra_indptr.tofile(os.path.join(out_dir, 'indptr.dat'))
    if verbose:
        print(f'[intra-csc] {total_intra:,}/{total_edges:,} edges survive '
              f'(intra ratio {total_intra / total_edges:.4f}) in one pass '
              f'({time.time() - t0:.1f}s)', flush=True)
    out_conf = {
        'num_nodes': num_nodes,
        'num_edges': total_intra,
        'indptr_shape': [num_nodes + 1], 'indptr_dtype': 'int64',
        'indices_shape': [total_intra], 'indices_dtype': 'int32',
        'csc_layout': 'symmetric',
        'intra_ratio': total_intra / total_edges,
        'src_csc_dir': os.path.abspath(csc_dir),
        'src_total_edges': total_edges,
        'part_id_path': os.path.abspath(part_id_path),
        'num_clusters': info.num_clusters,
    }
    json.dump(out_conf, open(os.path.join(out_dir, 'conf.json'), 'w'), indent=2)
    if verbose:
        print(f'[intra-csc] wrote {out_dir}: {total_intra:,} intra edges, ratio '
              f'{out_conf["intra_ratio"]:.4f}, indices int32 '
              f'({total_intra * 4 / 1e9:.1f} GB)', flush=True)
    return out_conf


def load_intra_csc(out_dir):
    """Load a precomputed intra CSC. Returns (indptr int64[N+1], indices int32 memmap, conf)."""
    conf = json.load(open(os.path.join(out_dir, 'conf.json')))
    indptr = np.fromfile(os.path.join(out_dir, 'indptr.dat'),
                         dtype=conf['indptr_dtype']).reshape(tuple(conf['indptr_shape']))
    indices = np.memmap(os.path.join(out_dir, 'indices.dat'), mode='r',
                        shape=tuple(conf['indices_shape']), dtype=conf['indices_dtype'])
    return indptr, indices, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['products', 'papers', 'friendster'], default=None)
    ap.add_argument('--part-k', type=int, default=1000, help='cluster count of the METIS partition')
    ap.add_argument('--csc-dir', default=None, help='override: dir with indptr.dat/indices.dat/conf.json')
    ap.add_argument('--part-id', default=None, help='override: part_id.pth path')
    ap.add_argument('--out-dir', default=None, help='override: output dir (default <part_dir>/intra_csc)')
    ap.add_argument('--block-edges', type=int, default=100_000_000)
    args = ap.parse_args()

    if args.csc_dir and args.part_id:
        csc_dir, part_id = args.csc_dir, args.part_id
    elif args.dataset:
        csc_dir, part_id = _dataset_paths(args.dataset, args.part_k)
    else:
        ap.error('pass --dataset, or both --csc-dir and --part-id')

    out_dir = args.out_dir or os.path.join(os.path.dirname(part_id), 'intra_csc')
    build_intra_csc(csc_dir, part_id, out_dir, block_edges=args.block_edges)


if __name__ == '__main__':
    main()
