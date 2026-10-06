"""Add one self-loop per node to an existing Ginex-layout CSC, in place of a rebuild.

prepare_dataset_sym.py can produce a self-looped CSC from the raw edge list, but on
an already-built dataset that is the expensive way round: it re-reads srcList/dstList,
re-scatters every entry into a scratch CSC (papers: +25.9 GB tmp) and re-runs the
per-node np.unique loop over 111M nodes. This script instead streams the canonical
CSC that is already on disk and splices the self entry into each column, which needs
only the output file (papers: 26.7 GB) and one vectorized pass.

The result is byte-identical to what prepare_dataset_sym.py writes with self-loops
on: exactly one self entry per node, columns still sorted ascending, and a column
that already held its self citation is left alone rather than doubled.

nc_score.pth is refreshed too, with the same formula prepare_dataset_sym.py uses, so a
dir stays usable by Ginex. Its neighbor cache proper -- nc_size_*.dat / nctbl_*.dat --
is built by Ginex_with_intra/create_neigh_cache.py, not by either of these scripts; the
run prints the exact command to rebuild whatever it finds. Nothing in FALCON reads any of
the three, so skipping them costs an FALCON-only workflow nothing.

Usage (from the repo root):
    python preprocess/add_self_loops.py --dataset papers
    python preprocess/add_self_loops.py --csc-dir <dir with indptr.dat/indices.dat/conf.json>
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys

import numpy as np
import torch
from tqdm import tqdm

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
from sampling.intra_edges import _dataset_paths                          # noqa: E402


def _row_blocks(indptr, block_edges):
    """Yield (r0, r1) row ranges each spanning ~block_edges entries."""
    n = indptr.shape[0] - 1
    r0 = 0
    while r0 < n:
        budget = int(indptr[r0]) + block_edges
        r1 = int(np.searchsorted(indptr, budget, side='right')) - 1
        r1 = min(max(r1, r0 + 1), n)          # always make progress, never overshoot
        yield r0, r1
        r0 = r1


def _check_sorted_unique(seg, rows):
    """Canonical CSC means each column is ascending with no repeats. Verify it.

    The splice below places the self entry by counting neighbors below the node id,
    which is only the right position if the column is sorted -- and only inserts at
    all if the column has no duplicate self entry. Both are cheap to confirm here and
    silently wrong to assume: an unsorted column would still produce a plausible file.
    """
    if seg.size < 2:
        return
    same = rows[1:] == rows[:-1]
    if not bool(np.all(seg[1:][same] > seg[:-1][same])):
        raise RuntimeError('source CSC column is not strictly ascending -- this script '
                           'assumes the canonical layout prepare_dataset_sym.py writes')


def _write_nc_score(csc_dir, mode, ref_count, col_degree):
    """nc_score.pth, byte-for-byte the formula prepare_dataset_sym.py writes.

    lib/cache.py ranks a node by out-neighbors / in-neighbors: how often its adjacency
    list is requested, over how much cache space it costs. Both counts are taken on the
    final graph, so the self entry is included on each side -- on a symmetric graph the
    ratio stays 1 everywhere, self-loops or not. Steers caching only, never accuracy.
    """
    if mode == 'degree':
        score_np = col_degree
    else:
        score_np = ref_count.astype(np.float32) / (col_degree + np.float32(1e-8))
    tmp = os.path.join(csc_dir, 'nc_score.pth.tmp')
    torch.save(torch.from_numpy(np.ascontiguousarray(score_np)), tmp)
    os.replace(tmp, os.path.join(csc_dir, 'nc_score.pth'))
    print(f'wrote nc_score.pth (score={mode})')


def _warn_stale_neighbor_cache(csc_dir):
    """The prebuilt Ginex neighbor cache is now keyed on a graph that no longer exists.

    Nothing deletes it here -- a half-updated dataset dir is worse than a stale file
    someone knowingly ignores -- but staying silent would let a Ginex run read it.
    """
    caches = sorted(glob.glob(os.path.join(csc_dir, 'nc_size_*.dat')))
    if not caches:
        return
    print('\nstale, built on the pre-self-loop graph:')
    for p in caches:
        print(f'  {os.path.basename(p)}')
    sizes = [m.group(1) for m in (re.search(r'nc_size_(\d+)\.dat$', p) for p in caches) if m]
    # create_neigh_cache.py builds its path as ./dataset/<--dataset>-ginex, so the flag
    # is the dir name minus that suffix -- not the short alias FALCON uses (papers/products).
    base = os.path.basename(os.path.normpath(csc_dir))
    name = base[:-len('-ginex')] if base.endswith('-ginex') else base
    print('FALCON never reads these. To refresh them for Ginex, from Ginex_with_intra/:')
    for s in sizes:
        print(f'  python create_neigh_cache.py --dataset {name} --neigh-cache-size {s}')


def refresh_nc_score(csc_dir, block_edges):
    """Recompute nc_score.pth from the CSC on disk, touching nothing else.

    For a dir whose graph was already converted -- add_self_loops refuses to run twice,
    but its nc_score.pth is still the pre-self-loop one.
    """
    conf = json.load(open(os.path.join(csc_dir, 'conf.json')))
    num_nodes = int(conf['num_nodes'])
    src_edges = int(conf['indices_shape'][0])
    indptr = np.fromfile(os.path.join(csc_dir, 'indptr.dat'),
                         dtype=conf['indptr_dtype']).astype(np.int64, copy=False)
    if int(indptr[-1]) != src_edges:
        raise RuntimeError(f'indptr total {int(indptr[-1])} != indices_shape {src_edges}')
    indices = np.memmap(os.path.join(csc_dir, 'indices.dat'), mode='r',
                        shape=(src_edges,), dtype=conf['indices_dtype'])

    print(f'source: {num_nodes:,} nodes, {src_edges:,} edges, '
          f"self_loop={conf.get('self_loop')}")
    ref_count = np.zeros(num_nodes, dtype=np.int64)
    starts = range(0, src_edges, block_edges)
    for e0 in tqdm(list(starts), desc='count', unit='block'):
        ref_count += np.bincount(np.asarray(indices[e0:e0 + block_edges]),
                                 minlength=num_nodes)
    col_degree = (indptr[1:] - indptr[:-1]).astype(np.float32)
    _write_nc_score(csc_dir, conf.get('nc_score', 'ginex'), ref_count, col_degree)
    _warn_stale_neighbor_cache(csc_dir)


def add_self_loops(csc_dir, block_edges, dry_run=False, nc_score=True):
    conf_path = os.path.join(csc_dir, 'conf.json')
    conf = json.load(open(conf_path))
    if conf.get('self_loop'):
        raise SystemExit(f'{conf_path} already says self_loop=true -- nothing to do. '
                         'Delete the key if you really want to rerun.')

    num_nodes = int(conf['num_nodes'])
    src_edges = int(conf['indices_shape'][0])
    idx_dtype = np.dtype(conf['indices_dtype'])

    indptr = np.fromfile(os.path.join(csc_dir, 'indptr.dat'),
                         dtype=conf['indptr_dtype']).astype(np.int64, copy=False)
    if indptr.shape[0] != num_nodes + 1:
        raise RuntimeError(f'indptr has {indptr.shape[0]} entries, expected {num_nodes + 1}')
    if int(indptr[-1]) != src_edges:
        raise RuntimeError(f'indptr total {int(indptr[-1])} != indices_shape {src_edges}')
    indices = np.memmap(os.path.join(csc_dir, 'indices.dat'), mode='r',
                        shape=(src_edges,), dtype=idx_dtype)

    # Upper bound: one added entry per node. The real count comes out of the pass and
    # the file is truncated to it, but the space has to be there up front.
    need = (src_edges + num_nodes) * idx_dtype.itemsize
    free = shutil.disk_usage(csc_dir).free
    print(f'source: {num_nodes:,} nodes, {src_edges:,} edges ({idx_dtype})')
    print(f'output needs <= {need / 1e9:.1f} GB, {free / 1e9:.1f} GB free on {csc_dir}')
    if need > free:
        raise SystemExit('not enough free space -- free some up, or point --csc-dir at '
                         'a copy on a roomier filesystem')
    if dry_run:
        print('--dry-run: stopping before writing')
        return

    # nc_score.pth exists only in dirs someone built for Ginex; don't invent one.
    want_score = nc_score and os.path.isfile(os.path.join(csc_dir, 'nc_score.pth'))
    score_mode = conf.get('nc_score', 'ginex')
    if nc_score and not want_score:
        print('no nc_score.pth here -- skipping it')

    out_indices_path = os.path.join(csc_dir, 'indices.dat.tmp')
    new_deg = np.empty(num_nodes, dtype=np.int64)
    # ref_count[v] = how many columns list v. Counted on the output, so self entries
    # are in it -- the same thing prepare_dataset_sym.py counts.
    ref_count = np.zeros(num_nodes, dtype=np.int64) if want_score else None
    added = 0

    with open(out_indices_path, 'wb') as out:
        for r0, r1 in tqdm(list(_row_blocks(indptr, block_edges)), desc='splice', unit='block'):
            e0, e1 = int(indptr[r0]), int(indptr[r1])
            seg = np.asarray(indices[e0:e1]).astype(np.int64, copy=False)
            deg = indptr[r0 + 1:r1 + 1] - indptr[r0:r1]
            nrows = r1 - r0
            rows = np.repeat(np.arange(r0, r1, dtype=np.int64), deg)
            _check_sorted_unique(seg, rows)

            local = rows - r0
            # A column keeps its length iff it already lists itself.
            has_self = np.zeros(nrows, dtype=bool)
            has_self[local[seg == rows]] = True
            insert = ~has_self
            # Sorted column -> the self entry belongs after every neighbor below it.
            n_below = np.bincount(local[seg < rows], minlength=nrows)

            out_deg = deg + insert
            out_off = np.empty(nrows + 1, dtype=np.int64)
            out_off[0] = 0
            np.cumsum(out_deg, out=out_off[1:])
            new_deg[r0:r1] = out_deg

            # Every old entry slides right by one iff a self entry is inserted ahead of it.
            pos_in_row = np.arange(e1 - e0, dtype=np.int64) - (indptr[rows] - e0)
            shifted = insert[local] & (pos_in_row >= n_below[local])
            dest = out_off[local] + pos_in_row + shifted

            block = np.empty(int(out_off[-1]), dtype=idx_dtype)
            block[dest] = seg
            self_rows = np.nonzero(insert)[0]
            block[out_off[self_rows] + n_below[self_rows]] = self_rows + r0
            added += self_rows.size
            if ref_count is not None:
                ref_count += np.bincount(block, minlength=num_nodes)

            out.write(block.tobytes())

    out_edges = src_edges + added
    new_indptr = np.empty(num_nodes + 1, dtype=np.int64)
    new_indptr[0] = 0
    np.cumsum(new_deg, out=new_indptr[1:])
    if int(new_indptr[-1]) != out_edges:
        raise RuntimeError(f'indptr total {int(new_indptr[-1])} != written {out_edges}')
    written = os.path.getsize(out_indices_path) // idx_dtype.itemsize
    if written != out_edges:
        raise RuntimeError(f'wrote {written} entries, expected {out_edges}')

    del indices
    indptr_tmp = os.path.join(csc_dir, 'indptr.dat.tmp')
    new_indptr.tofile(indptr_tmp)
    os.replace(out_indices_path, os.path.join(csc_dir, 'indices.dat'))
    os.replace(indptr_tmp, os.path.join(csc_dir, 'indptr.dat'))

    conf['indices_shape'] = [int(out_edges)]
    conf['indptr_shape'] = [num_nodes + 1]
    # Same key prepare_dataset_sym.py writes, so a dataset dir says which graph it holds
    # no matter which of the two built it.
    conf['self_loop'] = True
    conf_tmp = conf_path + '.tmp'
    json.dump(conf, open(conf_tmp, 'w'))
    os.replace(conf_tmp, conf_path)

    written_files = 'indices.dat, indptr.dat, conf.json'
    if want_score:
        _write_nc_score(csc_dir, score_mode, ref_count,
                        new_deg.astype(np.float32, copy=False))
        written_files += ', nc_score.pth'

    print(f'added {added:,} self-loops ({num_nodes - added:,} nodes already had one)')
    print(f'edges {src_edges:,} -> {out_edges:,}')
    print(f'updated {csc_dir}: {written_files}')
    _warn_stale_neighbor_cache(csc_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=['products', 'papers'], default=None)
    ap.add_argument('--csc-dir', default=None,
                    help='override: dir with indptr.dat/indices.dat/conf.json')
    ap.add_argument('--block-edges', type=int, default=100_000_000,
                    help='entries per pass; ~40 bytes of RAM each')
    ap.add_argument('--dry-run', action='store_true',
                    help='report sizes and free space, then stop')
    ap.add_argument('--skip-nc-score', action='store_true',
                    help="don't refresh nc_score.pth (Ginex neighbor-cache ranking). "
                         'Saves one bincount per block; FALCON never reads it.')
    ap.add_argument('--nc-score-only', action='store_true',
                    help='leave the graph alone; just recompute nc_score.pth from the '
                         'CSC on disk. For a dir already converted by an earlier run.')
    args = ap.parse_args()

    if args.csc_dir:
        csc_dir = args.csc_dir
    elif args.dataset:
        csc_dir, _ = _dataset_paths(args.dataset, 1000)   # part_k is irrelevant here
    else:
        ap.error('pass --dataset or --csc-dir')

    if args.nc_score_only:
        refresh_nc_score(csc_dir, args.block_edges)
        return

    add_self_loops(csc_dir, args.block_edges, dry_run=args.dry_run,
                   nc_score=not args.skip_nc_score)


if __name__ == '__main__':
    main()
