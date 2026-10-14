"""Convert an extracted OGB download into the flat binary layout prepare_dataset_sym.py reads.

This script does not download anything. Fetch the dataset from OGB yourself and unzip it
under data/ogb/ (any location works with --ogb-dir):

    mkdir -p data/ogb && cd data/ogb
    wget http://snap.stanford.edu/ogb/data/nodeproppred/products.zip        && unzip products.zip        # 1.4 GB -> products/
    wget http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip  && unzip papers100M-bin.zip  # 57 GB  -> papers100M-bin/

Input (the folder the zip extracts to):
    products/        raw/edge.csv.gz, node-feat.csv.gz, node-label.csv.gz; split/sales_ranking/*.csv.gz
    papers100M-bin/  raw/data.npz (edge_index, node_feat), raw/node-label.npz; split/time/*.csv.gz
Output (data/raw/<products|papers100M>/):
    srcList.bin, dstList.bin   int32 [E]      directed edge list
    feat.bin                   float32 [N*D]  node features, row-major
    labels.bin                 int64 [N]      class id, -1 where OGB has no label (NaN)
    trainIds.bin, valIds.bin, testIds.bin     int64, OGB's official split

ogbn-products  edge.csv.gz lists each undirected edge once; as OGB does, both directions
               are written. Everything fits in RAM (~3 GB).
ogbn-papers100M is too large to materialize (57 GB features + 26 GB edges), so the arrays
               are streamed straight out of the npz members without loading them.

Run (from the repo root):
    python preprocess/convert_ogb_raw.py --dataset ogbn-products
    python preprocess/prepare_dataset_sym.py --raw-dir data/raw/products --out-dir data/ogbn-products \\
        --num-features 100 --num-classes 47
    python preprocess/convert_ogb_raw.py --dataset ogbn-papers100M
    python preprocess/prepare_dataset_sym.py            # defaults are the papers100M paths
"""
import argparse
import gzip
import os
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib import format as npfmt

_ROOT = Path(__file__).resolve().parents[1]
OGB_URL = 'http://snap.stanford.edu/ogb/data/nodeproppred/'
DATASETS = {
    #  name              zip name              extracted folder    raw output folder  file that must exist
    'ogbn-products':   ('products.zip',       'products',         'products',        'raw/edge.csv.gz'),
    'ogbn-papers100M': ('papers100M-bin.zip', 'papers100M-bin',   'papers100M',      'raw/data.npz'),
}
CHUNK = 16_000_000          # elements per read when converting dtypes
COPY_BYTES = 64 << 20       # bytes per read when copying as-is


def write_atomic(path, write_fn):
    """Write through <path>.tmp and rename, so a killed run never leaves a truncated file."""
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'wb') as f:
        write_fn(f)
    os.replace(tmp, path)


def write_ids(path, ids):
    write_atomic(path, lambda f: np.ascontiguousarray(ids, dtype=np.int64).tofile(f))
    print(f'  {path.name}: {ids.size:,} ids', flush=True)


def write_labels(out, labels):
    labels = np.asarray(labels)
    if np.issubdtype(labels.dtype, np.floating):          # papers100M: NaN = unlabeled
        known = ~np.isnan(labels)
        out_lab = np.full(labels.shape, -1, dtype=np.int64)
        out_lab[known] = labels[known].astype(np.int64)
    else:
        out_lab = labels.astype(np.int64)
    write_atomic(out / 'labels.bin', lambda f: out_lab.tofile(f))
    print(f'  labels.bin: {out_lab.size:,} nodes, {int((out_lab >= 0).sum()):,} labeled, '
          f'{int(out_lab.max()) + 1} classes', flush=True)


def read_split_csv(path):
    with gzip.open(path, 'rt') as g:
        return np.loadtxt(g, dtype=np.int64, delimiter=',').reshape(-1)


# ---------------------------------------------------------------- ogbn-products (CSV, in RAM)
def convert_products(out, ogb_dir):
    import pandas as pd
    raw = ogb_dir / 'raw'
    print('reading edge.csv.gz', flush=True)
    edge = pd.read_csv(raw / 'edge.csv.gz', header=None, dtype=np.int64).values.T     # [2, E_undirected]
    # OGB stores each products edge once and adds the reverse at load time, interleaved
    # as (u,v),(v,u). Do the same so the directed list matches what OGB loaders produce.
    both = np.repeat(edge, 2, axis=1)
    both[0, 1::2] = edge[1]
    both[1, 1::2] = edge[0]
    if both.max() >= 2 ** 31:
        raise ValueError('node ids do not fit int32')
    print('reading node-feat.csv.gz (2.4M rows x 100)', flush=True)
    feat = pd.read_csv(raw / 'node-feat.csv.gz', header=None, dtype=np.float32).values
    labels = pd.read_csv(raw / 'node-label.csv.gz', header=None, dtype=np.int64).values.reshape(-1)
    if not (feat.shape[0] == labels.size > both.max()):
        raise RuntimeError(f'inconsistent sizes: feat {feat.shape}, labels {labels.shape}, max node id {both.max()}')
    print(f'nodes={feat.shape[0]:,} directed edges={both.shape[1]:,} feat={feat.shape}', flush=True)
    write_atomic(out / 'srcList.bin', lambda f: both[0].astype(np.int32).tofile(f))
    write_atomic(out / 'dstList.bin', lambda f: both[1].astype(np.int32).tofile(f))
    write_atomic(out / 'feat.bin', lambda f: np.ascontiguousarray(feat, dtype=np.float32).tofile(f))
    write_labels(out, labels)
    split_dir = ogb_dir / 'split' / 'sales_ranking'
    for csv, name in (('train.csv.gz', 'trainIds.bin'), ('valid.csv.gz', 'valIds.bin'), ('test.csv.gz', 'testIds.bin')):
        write_ids(out / name, read_split_csv(split_dir / csv))


# ---------------------------------------------------------------- ogbn-papers100M (npz, streamed)
def open_npy_member(npz_path, member):
    """Return (zipfile, stream positioned at the data, shape, dtype) for one array of an npz."""
    zf = zipfile.ZipFile(npz_path)
    f = zf.open(member)
    version = npfmt.read_magic(f)
    reader = {(1, 0): npfmt.read_array_header_1_0, (2, 0): npfmt.read_array_header_2_0}.get(version)
    if reader is None:
        raise RuntimeError(f'{member}: unsupported npy version {version}')
    shape, fortran, dtype = reader(f)
    if fortran:
        raise RuntimeError(f'{member} is Fortran-ordered')
    return zf, f, shape, np.dtype(dtype)


def stream_convert(f, count, src_dtype, dst_dtype, out_f, label):
    """Read `count` elements of src_dtype from f, write them as dst_dtype to out_f."""
    done = 0
    while done < count:
        n = min(CHUNK, count - done)
        buf = f.read(n * src_dtype.itemsize)
        if len(buf) != n * src_dtype.itemsize:
            raise RuntimeError(f'{label}: short read at element {done + len(buf) // src_dtype.itemsize:,}')
        arr = np.frombuffer(buf, dtype=src_dtype)
        if dst_dtype == np.int32 and arr.max(initial=0) >= 2 ** 31:
            raise ValueError(f'{label}: value does not fit int32')
        arr.astype(dst_dtype, copy=False).tofile(out_f)
        done += n
        print(f'  {label}: {done:,}/{count:,}', flush=True)


def stream_copy(f, nbytes, out_f, label):
    done = 0
    while done < nbytes:
        buf = f.read(min(COPY_BYTES, nbytes - done))
        if not buf:
            raise RuntimeError(f'{label}: short read at byte {done:,}')
        out_f.write(buf)
        done += len(buf)
        if done % (16 * COPY_BYTES) == 0 or done == nbytes:
            print(f'  {label}: {done / 2**30:.1f}/{nbytes / 2**30:.1f} GiB', flush=True)


def convert_papers(out, ogb_dir):
    data_npz = ogb_dir / 'raw' / 'data.npz'
    zf, f, shape, dtype = open_npy_member(data_npz, 'edge_index.npy')
    if len(shape) != 2 or shape[0] != 2 or dtype != np.int64:
        raise RuntimeError(f'unexpected edge_index: shape={shape} dtype={dtype}')
    num_edges = int(shape[1])
    print(f'directed edges={num_edges:,}', flush=True)
    # C-order [2, E]: the whole source row comes first, then the whole target row.
    write_atomic(out / 'srcList.bin', lambda o: stream_convert(f, num_edges, dtype, np.int32, o, 'srcList'))
    write_atomic(out / 'dstList.bin', lambda o: stream_convert(f, num_edges, dtype, np.int32, o, 'dstList'))
    f.close(); zf.close()

    zf, f, shape, dtype = open_npy_member(data_npz, 'node_feat.npy')
    print(f'node_feat shape={shape} dtype={dtype}', flush=True)
    count = int(np.prod(shape))
    if dtype == np.float32:
        write_atomic(out / 'feat.bin', lambda o: stream_copy(f, count * 4, o, 'feat'))
    else:
        write_atomic(out / 'feat.bin', lambda o: stream_convert(f, count, dtype, np.float32, o, 'feat'))
    f.close(); zf.close()

    labels = np.load(ogb_dir / 'raw' / 'node-label.npz')['node_label'].reshape(-1)
    write_labels(out, labels)
    split_dir = ogb_dir / 'split' / 'time'
    for csv, name in (('train.csv.gz', 'trainIds.bin'), ('valid.csv.gz', 'valIds.bin'), ('test.csv.gz', 'testIds.bin')):
        write_ids(out / name, read_split_csv(split_dir / csv))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=sorted(DATASETS), required=True)
    ap.add_argument('--ogb-dir', type=Path, default=None,
                    help='the extracted OGB folder (default: data/ogb/products or data/ogb/papers100M-bin)')
    ap.add_argument('--out-dir', type=Path, default=None,
                    help='default: data/raw/<products|papers100M>')
    args = ap.parse_args()
    zip_name, folder, raw_name, marker = DATASETS[args.dataset]
    ogb_dir = (args.ogb_dir or _ROOT / 'data' / 'ogb' / folder).resolve()
    if not (ogb_dir / marker).is_file():
        raise SystemExit(
            f'{ogb_dir / marker} not found.\n'
            f'Download {OGB_URL}{zip_name} and unzip it so that {ogb_dir} holds raw/ and split/, '
            f'or point --ogb-dir at the extracted folder.')
    out = (args.out_dir or _ROOT / 'data' / 'raw' / raw_name).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f'{args.dataset}: {ogb_dir} -> {out}', flush=True)
    if args.dataset == 'ogbn-products':
        convert_products(out, ogb_dir)
    else:
        convert_papers(out, ogb_dir)
    print('done', flush=True)


if __name__ == '__main__':
    main()
