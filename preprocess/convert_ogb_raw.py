"""Dump an OGB node-property dataset into the flat binary layout prepare_dataset_sym.py reads.

Output (data/raw/<products|papers100M>/):
    srcList.bin, dstList.bin   int32 [E]      directed edge list (OGB edge_index rows 0 and 1)
    feat.bin                   float32 [N*D]  node features, row-major
    labels.bin                 int64 [N]      class id, -1 where OGB has no label (NaN)
    trainIds.bin, valIds.bin, testIds.bin     int64, OGB's official split

ogbn-products  is loaded through the ogb package (it downloads products.zip into
               data/ogb/ and adds the reverse of every edge, as OGB specifies).
ogbn-papers100M is too large to materialize in RAM (57 GB features + 26 GB edges), so
               it is streamed straight out of the npz members of papers100M-bin.zip.
               The zip is downloaded into data/ogb/ on first use; a pre-extracted copy can
               be pointed at with --ogb-dir (the folder that holds raw/data.npz).

Next step (from the repo root):
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
    #  name            download zip          folder inside the zip   raw output folder
    'ogbn-products':   ('products.zip',       'products',            'products'),
    'ogbn-papers100M': ('papers100M-bin.zip', 'papers100M-bin',      'papers100M'),
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


# ---------------------------------------------------------------- ogbn-products (in RAM)
def convert_products(out, ogb_root):
    # ogb 1.3.6 caches the parsed graph with torch.save and reads it back with a bare
    # torch.load(); torch >= 2.6 defaults to weights_only=True, which rejects the numpy
    # arrays in that cache. The cache is written by this process from OGB's own CSVs.
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    from ogb.nodeproppred import NodePropPredDataset
    ds = NodePropPredDataset('ogbn-products', root=str(ogb_root))
    graph, labels = ds[0]
    split = ds.get_idx_split()
    ei = graph['edge_index']
    if ei.max() >= 2 ** 31:
        raise ValueError('node ids do not fit int32')
    print(f'nodes={graph["num_nodes"]:,} directed edges={ei.shape[1]:,} feat={graph["node_feat"].shape}', flush=True)
    write_atomic(out / 'srcList.bin', lambda f: ei[0].astype(np.int32).tofile(f))
    write_atomic(out / 'dstList.bin', lambda f: ei[1].astype(np.int32).tofile(f))
    write_atomic(out / 'feat.bin', lambda f: np.ascontiguousarray(graph['node_feat'], dtype=np.float32).tofile(f))
    write_labels(out, labels.reshape(-1))
    for key, name in (('train', 'trainIds.bin'), ('valid', 'valIds.bin'), ('test', 'testIds.bin')):
        write_ids(out / name, split[key])


# ---------------------------------------------------------------- ogbn-papers100M (streamed)
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
        with gzip.open(split_dir / csv, 'rt') as g:
            write_ids(out / name, np.loadtxt(g, dtype=np.int64, delimiter=',').reshape(-1))


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


def ensure_download(ogb_root, zip_name, folder):
    """Download and extract <zip_name> into ogb_root unless <ogb_root>/<folder> already exists."""
    target = ogb_root / folder
    if target.is_dir():
        return target
    from ogb.utils.url import download_url, extract_zip
    ogb_root.mkdir(parents=True, exist_ok=True)
    path = download_url(OGB_URL + zip_name, str(ogb_root))
    extract_zip(path, str(ogb_root))
    os.unlink(path)
    if not target.is_dir():
        raise RuntimeError(f'{zip_name} did not extract to {target}')
    return target


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=sorted(DATASETS), required=True)
    ap.add_argument('--ogb-root', type=Path, default=_ROOT / 'data' / 'ogb',
                    help='where OGB downloads are kept (default: data/ogb)')
    ap.add_argument('--ogb-dir', type=Path, default=None,
                    help='ogbn-papers100M only: an already extracted papers100M-bin folder '
                         '(holds raw/data.npz); skips the download')
    ap.add_argument('--out-dir', type=Path, default=None,
                    help='default: data/raw/<products|papers100M>')
    args = ap.parse_args()
    zip_name, folder, raw_name = DATASETS[args.dataset]
    out = (args.out_dir or _ROOT / 'data' / 'raw' / raw_name).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f'{args.dataset} -> {out}', flush=True)
    if args.dataset == 'ogbn-products':
        convert_products(out, args.ogb_root.resolve())
    else:
        ogb_dir = args.ogb_dir.resolve() if args.ogb_dir else ensure_download(args.ogb_root.resolve(), zip_name, folder)
        if not (ogb_dir / 'raw' / 'data.npz').is_file():
            raise FileNotFoundError(f'{ogb_dir}/raw/data.npz not found')
        convert_papers(out, ogb_dir)
    print('done', flush=True)


if __name__ == '__main__':
    main()
