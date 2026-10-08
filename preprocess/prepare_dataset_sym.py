"""Build a Ginex dataset directory, symmetrizing the graph by default.

Differs from Ginex's original dataset builder in two things.

First, every raw directed edge (s -> d)
also contributes a reverse entry, so the CSC column of a node holds both the
papers citing it and the papers it cites. The raw ogbn-papers100M edge list is
directed, and the splits are temporal -- test papers are the newest ones, which
almost nothing cites yet. Keeping only in-edges leaves 61% of test nodes with no
neighbors at all, so SAGE degenerates to an MLP over their own features there and
test accuracy collapses (measured: 0.462 vs the ~0.659 reported for Ginex).
Reverse edges are a paper's own reference list -- known at publication time, so
this leaks nothing. DiskGNN (arXiv:2405.05231 §7.1) does the same: "For PS, we add
a reverse edge for each directed edge to enlarge the receptive field of each node
during neighbor aggregation", and reports 3.3B edges for papers100M.

Second, every node gets exactly one self-loop, so a fanout sample of a node can
draw the node's own features and an isolated node still aggregates something
instead of producing a zero neighbor sum. DiskGNN's loader does the same --
its loader runs remove_self_loop then add_self_loop -- so the two systems walk
the same adjacency. "Exactly one" matters: the raw list already contains a few
self citations, and those are collapsed rather than doubled.

Pass --no-symmetrize / --no-self-loop to reproduce the original behavior.
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


DEFAULT_RAW_DIR = Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'papers100M'
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / 'data' / 'ogbn-papers100M-sym'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--raw-dir', type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument('--tmp-dir', type=Path, default=Path('/tmp'))
    parser.add_argument('--num-features', type=int, default=128)
    parser.add_argument('--num-classes', type=int, default=172)
    parser.add_argument('--chunk-edges', type=int, default=8_000_000)
    parser.add_argument('--keep-tmp', action='store_true')
    parser.add_argument('--no-symmetrize', dest='symmetrize', action='store_false',
                        help='keep the raw directed CSC, as Ginex\'s original builder does.')
    parser.add_argument('--no-self-loop', dest='self_loop', action='store_false',
                        help='do not add a self-loop to every node, as '
                             'Ginex\'s original builder does.')
    parser.add_argument('--score', choices=['ginex', 'degree'], default='ginex',
                        help="nc_score.pth formula. 'ginex' is upstream's "
                             'out-degree/in-degree; on a symmetric graph the two are '
                             "equal so it collapses to 1 everywhere. 'degree' ranks by "
                             'neighbor count instead, which keeps the neighbor cache '
                             'meaningful but deviates from upstream. Affects cache hit '
                             'rate (speed) only, never accuracy.')
    parser.add_argument('--graph-only', action='store_true',
                        help='rebuild indptr/indices/nc_score only, reusing the existing '
                             'features.dat, labels.dat and split_idx.pth in --out-dir. '
                             'Those three do not depend on edge direction.')
    return parser.parse_args()


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(path)


def file_numel(path, dtype):
    return path.stat().st_size // np.dtype(dtype).itemsize


def copy_memmap(src_path, dst_path, shape, dtype, chunk_rows):
    src = np.memmap(src_path, mode='r', shape=shape, dtype=dtype)
    dst_tmp = dst_path.with_suffix(dst_path.suffix + '.tmp')
    dst = np.memmap(dst_tmp, mode='w+', shape=shape, dtype=dtype)
    for start in tqdm(range(0, shape[0], chunk_rows), desc=f'copy {src_path.name}', unit='chunk'):
        end = min(start + chunk_rows, shape[0])
        dst[start:end] = src[start:end]
    dst.flush()
    del src, dst
    os.replace(dst_tmp, dst_path)


def write_labels(src_path, dst_path, num_nodes, chunk_rows):
    src = np.memmap(src_path, mode='r', shape=(num_nodes,), dtype=np.int64)
    dst_tmp = dst_path.with_suffix(dst_path.suffix + '.tmp')
    dst = np.memmap(dst_tmp, mode='w+', shape=(num_nodes,), dtype=np.float32)
    for start in tqdm(range(0, num_nodes, chunk_rows), desc='copy labels', unit='chunk'):
        end = min(start + chunk_rows, num_nodes)
        dst[start:end] = src[start:end].astype(np.float32, copy=False)
    dst.flush()
    del src, dst
    os.replace(dst_tmp, dst_path)


def save_split(raw_dir, out_path):
    split_idx = {
        'train': torch.from_numpy(np.fromfile(raw_dir / 'trainIds.bin', dtype=np.int64)),
        'valid': torch.from_numpy(np.fromfile(raw_dir / 'valIds.bin', dtype=np.int64)),
        'test': torch.from_numpy(np.fromfile(raw_dir / 'testIds.bin', dtype=np.int64)),
    }
    torch.save(split_idx, out_path)
    print(
        'split sizes: '
        f"train={split_idx['train'].numel()} "
        f"valid={split_idx['valid'].numel()} "
        f"test={split_idx['test'].numel()}",
        flush=True,
    )


def _edge_chunks(src, dst, num_edges, chunk_edges, symmetrize):
    """Yield (col, val) int64 chunks describing CSC entries to write.

    A raw edge (s -> d) places s in column d. When symmetrizing it also places d
    in column s, which is what turns the citation graph undirected.
    """
    for start in range(0, num_edges, chunk_edges):
        end = min(start + chunk_edges, num_edges)
        chunk_src = src[start:end].astype(np.int64, copy=False)
        chunk_dst = dst[start:end].astype(np.int64, copy=False)
        yield chunk_dst, chunk_src
        if symmetrize:
            yield chunk_src, chunk_dst


def _num_chunks(num_edges, chunk_edges, symmetrize):
    n = (num_edges + chunk_edges - 1) // chunk_edges
    return n * (2 if symmetrize else 1)


def build_degree_and_csc(raw_dir, out_dir, tmp_dir, num_nodes, num_edges, chunk_edges,
                         symmetrize, self_loop, score_mode):
    src_path = raw_dir / 'srcList.bin'
    dst_path = raw_dir / 'dstList.bin'

    src = np.memmap(src_path, mode='r', shape=(num_edges,), dtype=np.int32)
    dst = np.memmap(dst_path, mode='r', shape=(num_edges,), dtype=np.int32)

    # Directed degrees of the raw edge list, kept separately from the CSC column
    # degree because upstream's neighbor-cache score is defined on them.
    dir_in = np.memmap(tmp_dir / 'dir_in_degree.dat', mode='w+', shape=(num_nodes,), dtype=np.int64)
    dir_out = np.memmap(tmp_dir / 'dir_out_degree.dat', mode='w+', shape=(num_nodes,), dtype=np.int64)
    ref_count = np.memmap(tmp_dir / 'ref_count.dat', mode='w+', shape=(num_nodes,), dtype=np.int64)
    dir_in[:] = 0
    dir_out[:] = 0
    ref_count[:] = 0

    print('Counting raw degrees...', flush=True)
    for start in tqdm(range(0, num_edges, chunk_edges), desc='degree', unit='chunk'):
        end = min(start + chunk_edges, num_edges)
        np.add.at(dir_in, dst[start:end].astype(np.int64, copy=False), 1)
        np.add.at(dir_out, src[start:end].astype(np.int64, copy=False), 1)
    dir_in.flush()
    dir_out.flush()

    # Column degree of the CSC about to be written. Symmetrizing adds a reverse
    # entry per edge, so each column holds in + out before dedup.
    total_entries = num_edges * (2 if symmetrize else 1)
    raw_indptr = np.empty(num_nodes + 1, dtype=np.int64)
    raw_indptr[0] = 0
    if symmetrize:
        np.cumsum(np.asarray(dir_in) + np.asarray(dir_out), out=raw_indptr[1:])
    else:
        np.cumsum(dir_in, out=raw_indptr[1:])
    if int(raw_indptr[-1]) != total_entries:
        raise RuntimeError(f'raw indptr entry count mismatch: {raw_indptr[-1]} != {total_entries}')

    write_pos = np.memmap(tmp_dir / 'write_pos.dat', mode='w+', shape=(num_nodes,), dtype=np.int64)
    write_pos[:] = raw_indptr[:-1]
    write_pos.flush()

    raw_indices_tmp = tmp_dir / 'indices_raw_csc.dat'
    raw_indices = np.memmap(raw_indices_tmp, mode='w+', shape=(total_entries,), dtype=np.int64)

    print(f'Writing raw CSC entries ({"symmetric" if symmetrize else "directed"})...', flush=True)
    chunks = _edge_chunks(src, dst, num_edges, chunk_edges, symmetrize)
    for col, val in tqdm(chunks, desc='raw entries', unit='chunk',
                         total=_num_chunks(num_edges, chunk_edges, symmetrize)):
        order = np.argsort(col, kind='stable')
        sorted_col = col[order]
        sorted_val = val[order]
        unique_col, first, counts = np.unique(sorted_col, return_index=True, return_counts=True)

        for node, offset, count in zip(unique_col, first, counts):
            pos = write_pos[node]
            raw_indices[pos:pos + count] = sorted_val[offset:offset + count]
            write_pos[node] = pos + count

        raw_indices.flush()
        write_pos.flush()

    if not np.array_equal(write_pos, raw_indptr[1:]):
        raise RuntimeError('Raw CSC write positions do not match raw indptr')

    print('Canonicalizing CSC like scipy coo.tocsc()...', flush=True)
    canonical_indices_tmp = out_dir / 'indices.dat.tmp'
    # A self-loop adds at most one entry per column on top of the raw entries, so the
    # scratch file needs that much headroom. It is truncated to the real size below.
    capacity = total_entries + (num_nodes if self_loop else 0)
    canonical_indices = np.memmap(canonical_indices_tmp, mode='w+', shape=(capacity,), dtype=np.int64)
    canonical_indptr = np.empty(num_nodes + 1, dtype=np.int64)
    canonical_indptr[0] = 0

    # np.unique here also collapses the duplicate a node picks up when a citation is
    # reciprocal (both s -> d and d -> s exist), so the edge count lands below 2x.
    nonzero_nodes = np.flatnonzero(raw_indptr[1:] - raw_indptr[:-1])
    # Skipping empty columns is only valid without self-loops. With them an isolated
    # node still owns one entry, so every node has to go through the loop.
    nodes_to_write = np.arange(num_nodes) if self_loop else nonzero_nodes
    write_cursor = 0
    next_indptr_slot = 0
    for node in tqdm(nodes_to_write, desc='canonical csc', unit='node'):
        node = int(node)
        canonical_indptr[next_indptr_slot:node + 1] = write_cursor

        start = int(raw_indptr[node])
        end = int(raw_indptr[node + 1])
        neighbors = np.unique(raw_indices[start:end])
        if self_loop:
            # np.unique returns sorted, so inserting at the searchsorted position keeps
            # the column sorted, and the equality check drops the duplicate when the
            # raw list already held the self edge -- exactly one self-loop either way.
            at = int(np.searchsorted(neighbors, node))
            if at == neighbors.size or int(neighbors[at]) != node:
                neighbors = np.insert(neighbors, at, node)
        count = neighbors.size
        canonical_indices[write_cursor:write_cursor + count] = neighbors
        np.add.at(ref_count, neighbors, 1)
        write_cursor += count
        canonical_indptr[node + 1] = write_cursor
        next_indptr_slot = node + 2

    canonical_indptr[next_indptr_slot:] = write_cursor
    canonical_edges = int(write_cursor)
    canonical_indices.flush()
    ref_count.flush()

    del canonical_indices
    with open(canonical_indices_tmp, 'r+b') as f:
        f.truncate(canonical_edges * np.dtype(np.int64).itemsize)
    os.replace(canonical_indices_tmp, out_dir / 'indices.dat')

    indptr_tmp = out_dir / 'indptr.dat.tmp'
    canonical_indptr.tofile(indptr_tmp)
    os.replace(indptr_tmp, out_dir / 'indptr.dat')

    # Ginex's NeighborCache (lib/cache.py in Ginex) scores a node by out-neighbors / in-neighbors: how
    # often its adjacency list gets requested, over how much cache space it costs.
    # On a symmetric graph those are the same number, so the ratio is 1 everywhere
    # and the ranking carries no signal -- hence --score degree as an alternative.
    # Either way this only steers which lists get cached, never what is computed.
    col_degree = (canonical_indptr[1:] - canonical_indptr[:-1]).astype(np.float32)
    if score_mode == 'degree':
        score_np = col_degree
    else:
        score_np = ref_count[:].astype(np.float32) / (col_degree + np.float32(1e-8))
    score = torch.from_numpy(np.ascontiguousarray(score_np))
    torch.save(score, out_dir / 'nc_score.pth')

    print(f'Saved canonical CSC with {canonical_edges} edges '
          f'({canonical_edges / max(1, num_edges):.2f}x the raw {num_edges})', flush=True)
    if symmetrize:
        # Counted before self-loops, so this still answers "how many nodes have a real
        # neighbor" -- with self-loops the raw column count would trivially be 100%.
        nonzero_frac = 100.0 * nonzero_nodes.size / num_nodes
        print(f'  nodes with >=1 real neighbor: {nonzero_nodes.size:,} '
              f'({nonzero_frac:.1f}%)', flush=True)
    if self_loop:
        print(f'  self-loops: one per node ({num_nodes:,} entries incl. '
              f'{num_nodes - nonzero_nodes.size:,} isolated nodes)', flush=True)
    print(f'Saved nc_score.pth (score={score_mode})', flush=True)

    del src, dst, dir_in, dir_out, ref_count, write_pos, raw_indices
    return canonical_indptr, canonical_edges


def main():
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    out_dir = args.out_dir.resolve()
    tmp_root = args.tmp_dir.resolve()

    for name in [
        'srcList.bin',
        'dstList.bin',
        'feat.bin',
        'labels.bin',
        'trainIds.bin',
        'valIds.bin',
        'testIds.bin',
    ]:
        require_file(raw_dir / name)

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix='prepare_dataset_sym_', dir=tmp_root))

    try:
        num_edges = file_numel(raw_dir / 'srcList.bin', np.int32)
        dst_edges = file_numel(raw_dir / 'dstList.bin', np.int32)
        if num_edges != dst_edges:
            raise RuntimeError(f'src/dst edge count mismatch: {num_edges} != {dst_edges}')

        feature_numel = file_numel(raw_dir / 'feat.bin', np.float32)
        if feature_numel % args.num_features != 0:
            raise RuntimeError(f'feature file is not divisible by {args.num_features}')
        num_nodes = feature_numel // args.num_features
        label_numel = file_numel(raw_dir / 'labels.bin', np.int64)
        if label_numel != num_nodes:
            raise RuntimeError(f'label/node count mismatch: {label_numel} != {num_nodes}')

        print(f'raw_dir={raw_dir}', flush=True)
        print(f'out_dir={out_dir}', flush=True)
        print(f'tmp_dir={tmp_dir}', flush=True)
        print(f'num_nodes={num_nodes} num_edges={num_edges}', flush=True)
        print(f'symmetrize={args.symmetrize} self_loop={args.self_loop} '
              f'score={args.score} graph_only={args.graph_only}', flush=True)

        if args.graph_only:
            # features/labels/split are independent of edge direction, so a rerun that
            # only fixes the graph can keep them -- but only if they are actually there.
            for name in ['features.dat', 'labels.dat', 'split_idx.pth']:
                if not (out_dir / name).is_file():
                    raise FileNotFoundError(
                        f'--graph-only needs an existing {name} in {out_dir}')

        indptr, canonical_edges = build_degree_and_csc(
            raw_dir, out_dir, tmp_dir, num_nodes, num_edges, args.chunk_edges,
            args.symmetrize, args.self_loop, args.score)

        if args.graph_only:
            print('Reusing existing features.dat / labels.dat / split_idx.pth', flush=True)
        else:
            print('Saving features.dat...', flush=True)
            copy_memmap(
                raw_dir / 'feat.bin',
                out_dir / 'features.dat',
                (num_nodes, args.num_features),
                np.float32,
                chunk_rows=max(1, args.chunk_edges // args.num_features),
            )

            print('Saving labels.dat...', flush=True)
            write_labels(raw_dir / 'labels.bin', out_dir / 'labels.dat', num_nodes, args.chunk_edges)

            print('Saving split_idx.pth...', flush=True)
            save_split(raw_dir, out_dir / 'split_idx.pth')

        conf = {
            'num_nodes': int(num_nodes),
            'indptr_shape': tuple(indptr.shape),
            'indptr_dtype': str(indptr.dtype),
            'indices_shape': (int(canonical_edges),),
            'indices_dtype': 'int64',
            'features_shape': (int(num_nodes), int(args.num_features)),
            'features_dtype': 'float32',
            'labels_shape': (int(num_nodes),),
            'labels_dtype': 'float32',
            'num_classes': int(args.num_classes),
            # Same key the part_id conf.json files use, so it is obvious at a glance
            # whether a dataset and a partition were built on the same graph.
            'csc_layout': 'symmetric' if args.symmetrize else 'directed',
            # A dataset dir has to say whether its columns carry self-loops -- otherwise
            # a rebuild silently changes what the sampler walks, with no way to tell.
            'self_loop': bool(args.self_loop),
            'raw_directed_edges': int(num_edges),
            'nc_score': args.score,
        }
        conf_tmp = out_dir / 'conf.json.tmp'
        with open(conf_tmp, 'w') as f:
            json.dump(conf, f)
        os.replace(conf_tmp, out_dir / 'conf.json')
        print('Done!', flush=True)

    except Exception:
        print(f'Failed. Temp files kept at: {tmp_dir}', flush=True)
        raise
    else:
        if args.keep_tmp:
            print(f'Keeping temp files at: {tmp_dir}', flush=True)
        else:
            shutil.rmtree(tmp_dir)
            print('Removed temp files.', flush=True)


if __name__ == '__main__':
    main()
