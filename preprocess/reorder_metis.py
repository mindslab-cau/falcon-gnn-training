"""Offline METIS node reordering; source files are opened read-only.

Run from the repo root: python -m preprocess.reorder_metis --dataset products --part-k 1000
Output defaults to data/reordered/<dataset>-k<k>. Existing outputs are refused.
O(N) ID/pointer arrays live in RAM; feature and edge copies use bounded blocks.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from sampling.intra_edges import _dataset_paths


def read_array(path, dtype, shape):
    dtype = np.dtype(dtype)
    if Path(path).stat().st_size != int(np.prod(shape)) * dtype.itemsize:
        raise ValueError(f'Unexpected file size: {path}')
    if not np.prod(shape):
        return np.empty(shape, dtype=dtype)
    return np.memmap(path, mode='r', dtype=dtype, shape=tuple(shape))


def reorder(csc_dir, part_path, out_dir, block_rows=65536, block_edges=1000000):
    src, part_path, dst = map(lambda p: Path(p).resolve(), (csc_dir, part_path, out_dir))
    if block_rows <= 0 or block_edges <= 0:
        raise ValueError('Block sizes must be positive')
    if dst == src or src in dst.parents or dst in src.parents:
        raise ValueError('Output must be separate from the source directory')
    if dst.exists():
        raise FileExistsError(f'Refusing to overwrite {dst}')
    conf = json.loads((src / 'conf.json').read_text())
    n, f = map(int, conf['features_shape'])
    if n <= 0 or int(conf.get('num_nodes', n)) != n:
        raise ValueError('Invalid node count')
    part = torch.load(part_path, map_location='cpu', weights_only=True)
    part = np.asarray(part)
    if part.shape != (n,) or part.dtype.kind not in 'iu' or np.any(part < 0):
        raise ValueError('Expected nonnegative integer partition IDs of shape [N]')
    features = read_array(src / 'features.dat', conf['features_dtype'], (n, f))
    label_shape = tuple(conf.get('labels_shape', [n]))
    if label_shape[0] != n:
        raise ValueError('Labels must have N rows')
    labels = read_array(src / 'labels.dat', conf['labels_dtype'], label_shape)
    ptr = read_array(src / 'indptr.dat', conf['indptr_dtype'], (n + 1,))
    if ptr.dtype.kind not in 'iu' or ptr[0] != 0 or np.any(ptr[1:] < ptr[:-1]):
        raise ValueError('Invalid CSC indptr')
    m = int(ptr[-1])
    edges = read_array(src / 'indices.dat', conf['indices_dtype'], (m,))
    if edges.dtype.kind not in 'iu':
        raise ValueError('CSC indices must be integers')
    for start in range(0, m, block_edges):
        e = edges[start:start + block_edges]
        if np.any(e < 0) or np.any(e >= n):
            raise ValueError('CSC neighbor ID out of range')
    splits = torch.load(src / 'split_idx.pth', map_location='cpu', weights_only=True)
    for key, ids in splits.items():
        a = np.asarray(ids)
        if a.dtype.kind not in 'iu' or np.any(a < 0) or np.any(a >= n):
            raise ValueError(f'Invalid split IDs: {key}')

    print(f'Sorting {n:,} nodes by (partition, original ID)', flush=True)
    new_to_old = np.argsort(part, kind='stable').astype(np.int64, copy=False)
    old_to_new = np.empty(n, dtype=np.int64)
    old_to_new[new_to_old] = np.arange(n, dtype=np.int64)
    new_ptr = np.empty(n + 1, dtype=np.int64)
    new_ptr[0] = 0
    # Bounded temporary arrays even when the graph has hundreds of millions of nodes.
    for start in range(0, n, block_rows):
        stop = min(n, start + block_rows)
        old = new_to_old[start:stop]
        new_ptr[start + 1:stop + 1] = ptr[old + 1] - ptr[old]
    np.cumsum(new_ptr, out=new_ptr)
    if m > np.iinfo(ptr.dtype).max or n - 1 > np.iinfo(edges.dtype).max:
        raise ValueError('Source index dtype cannot represent reordered graph')

    dst.mkdir(parents=True, exist_ok=False)
    (dst / 'INCOMPLETE').write_text('Only use this dataset after successful completion.\n')
    started = time.time()
    np.save(dst / 'old_to_new.npy', old_to_new)
    np.save(dst / 'new_to_old.npy', new_to_old)
    sorted_part = part[new_to_old]
    torch.save(torch.from_numpy(sorted_part.copy()), dst / 'part_id.pth')
    boundaries = np.r_[0, np.flatnonzero(sorted_part[1:] != sorted_part[:-1]) + 1, n]
    np.save(dst / 'cluster_offsets.npy', boundaries.astype(np.int64))
    np.save(dst / 'cluster_ids.npy', sorted_part[boundaries[:-1]])
    del sorted_part

    for name, source in [('features.dat', features), ('labels.dat', labels)]:
        print(f'Writing {name}', flush=True)
        with (dst / name).open('wb') as out:
            for start in range(0, n, block_rows):
                source[new_to_old[start:start + block_rows]].tofile(out)

    print(f'Writing CSC ({m:,} edges)', flush=True)
    new_ptr.astype(ptr.dtype, copy=False).tofile(dst / 'indptr.dat')
    # Translate bounded ranges of output edge positions to original CSC positions.
    # This preserves each adjacency list's order, multiplicity and self-loops.
    with (dst / 'indices.dat').open('wb') as out:
        for start in range(0, m, block_edges):
            pos = np.arange(start, min(m, start + block_edges), dtype=np.int64)
            rows = np.searchsorted(new_ptr, pos, side='right') - 1
            old_pos = ptr[new_to_old[rows]].astype(np.int64) + pos - new_ptr[rows]
            old_to_new[edges[old_pos]].astype(edges.dtype, copy=False).tofile(out)
    torch.save({key: torch.from_numpy(old_to_new[np.asarray(ids)].copy())
                for key, ids in splits.items()}, dst / 'split_idx.pth')
    conf.update(num_nodes=n, num_edges=m, indptr_shape=[n + 1], indices_shape=[m],
                features_shape=[n, f], labels_shape=list(label_shape),
                node_order='metis_cluster_then_original_id')
    (dst / 'conf.json').write_text(json.dumps(conf, indent=2) + '\n')
    metadata = dict(source_dir=str(src), source_partition=str(part_path),
                    num_nodes=n, num_edges=m, num_clusters=len(boundaries) - 1,
                    block_rows=block_rows, block_edges=block_edges,
                    elapsed_sec=time.time() - started,
                    mapping='new_to_old[new_id] = old_id; old_to_new[old_id] = new_id',
                    cluster_ranges='cluster_ids[i]: [cluster_offsets[i], cluster_offsets[i+1])')
    (dst / 'reorder_meta.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (dst / 'INCOMPLETE').unlink()
    print(f'Complete: {dst}', flush=True)
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', choices=['products', 'papers', 'friendster'], default='products')
    ap.add_argument('--part-k', type=int, default=1000)
    ap.add_argument('--csc-dir', help='Override source dataset directory')
    ap.add_argument('--part-id', help='Override METIS part_id.pth')
    ap.add_argument('--out-dir', help='New directory; existing directories are never overwritten')
    ap.add_argument('--block-rows', type=int, default=65536)
    ap.add_argument('--block-edges', type=int, default=1000000)
    args = ap.parse_args()
    csc, part = _dataset_paths(args.dataset, args.part_k)
    dst = args.out_dir or str(Path(__file__).resolve().parents[1] / 'data' / 'reordered'
                             / f'{args.dataset}-k{args.part_k}')
    reorder(args.csc_dir or csc, args.part_id or part, dst,
            args.block_rows, args.block_edges)


if __name__ == '__main__':
    main()
