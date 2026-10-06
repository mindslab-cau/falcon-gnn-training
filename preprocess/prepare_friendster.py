"""Build a Ginex-layout dataset dir for SNAP com-friendster, ready for FALCON.

Output is the same layout ogbn-papers100M-sym uses, so everything that goes
through sampling.intra_edges._dataset_paths works on it unchanged:

    conf.json  indptr.dat (int64)  indices.dat (int64)  features.dat (float32 N x D)
    labels.dat (float32)  split_idx.pth {train, valid, test}  nc_score.pth
    snap_ids.npy (int32, new id -> SNAP id)

Graph. The SNAP file lists each undirected edge once. Every edge is written into both
endpoint columns, every column gets exactly one self-loop, duplicates are collapsed and
each column is sorted -- the canonical form prepare_dataset_sym.py produces for papers
(csc_layout "symmetric", self_loop true). SNAP ids are sparse (max ~124.8M for 65.6M
nodes), so they are compacted to 0..N-1 in ascending SNAP-id order.

Features / labels. Friendster has neither. As DiskGNN does (examples/load_graph.py,
load_friendster(root, 128, 20)), features are uniform [0,1) float32 and labels are
uniform random over num_classes. Both are seeded, so a rebuild is byte-identical.
Accuracy on these labels is chance level by construction -- the dataset is for speed,
I/O and memory comparisons only.

Stages (each can be run alone; `all` runs them in order):
    plan    print disk/RAM needs against what is free, write nothing
    edges   pigz -dc | parallel parse -> int32 src/dst in --tmp-dir, SNAP ids compacted
    csc     bucketed sort/dedup -> indptr.dat, indices.dat, nc_score.pth
            (deletes the tmp edge files on success unless --keep-tmp)
    feat    features.dat, labels.dat, split_idx.pth
    verify  sizes, ranges, split disjointness, sampled symmetry / sortedness / self-loops

Resources (measured sizes of the final files; RAM is the peak of the stage):
    edges   tmp disk 14.4 GB             RAM ~8 GB (parse window)
    csc     indices 29.4 GB + indptr 0.5 GB, needs the 14.4 GB tmp alongside
                                         RAM ~36 GB (all CSC keys resident as int64)
    feat    features 31.4 GB (D=128) + labels 0.26 GB   RAM ~2 GB
    final   ~62 GB in --out-dir

After this, partition and register (see the printout at the end of `all`):
    python preprocess/do_metis_new.py --dataset friendster --metis-k 1000 \\
        --csc-dir data/friendster/sym --validate-symmetry

Usage (from the repo root):
    python preprocess/prepare_friendster.py --stage plan
    python preprocess/prepare_friendster.py --stage all
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


_REPO = Path(__file__).resolve().parents[1]
DEFAULT_RAW = _REPO / 'data' / 'friendster' / 'raw' / 'com-friendster.ungraph.txt.gz'
DEFAULT_OUT = _REPO / 'data' / 'friendster' / 'sym'
DEFAULT_TMP = _REPO / 'data' / 'friendster' / 'tmp'

# Fixed, not a flag: the random stream depends on how rows are chunked, and a
# rebuild with a different chunk size must not silently produce different features.
_FEAT_CHUNK_ROWS = 1 << 22

# Expected from the SNAP header; used only by `plan`, the real run reads the header.
_SNAP_NODES = 65_608_366
_SNAP_EDGES = 1_806_067_135

_GB = 1e9


# ----------------------------------------------------------------------------- utils

def _write_json(path, obj):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    os.replace(tmp, path)


def _update_conf(out_dir, **keys):
    """Each stage owns its own keys; merge them so stages can run separately."""
    path = out_dir / 'conf.json'
    conf = json.loads(path.read_text()) if path.is_file() else {}
    conf.update(keys)
    _write_json(path, conf)
    return conf


def _need_disk(path, need_bytes, what):
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    print(f'[disk] {what}: needs {need_bytes / _GB:.1f} GB, {free / _GB:.1f} GB free on {path}',
          flush=True)
    if need_bytes > free:
        raise SystemExit(f'not enough disk for {what} -- free '
                         f'{(need_bytes - free) / _GB:.1f} GB more and rerun this stage')


def _edges_meta_path(tmp_dir):
    return tmp_dir / 'edges_meta.json'


# ----------------------------------------------------------------------- stage: plan

def stage_plan(args):
    n, e = _SNAP_NODES, _SNAP_EDGES
    m = 2 * e + n                         # upper bound: SNAP has no duplicate/self edges
    rows = [
        ('tmp src/dst int32 (edges -> csc)', 2 * e * 4, args.tmp_dir),
        ('indices.dat int64', m * 8, args.out_dir),
        ('indptr.dat int64', (n + 1) * 8, args.out_dir),
        (f'features.dat float32 x{args.feat_dim}', n * args.feat_dim * 4, args.out_dir),
        ('labels.dat + nc_score + snap_ids', n * 4 * 3, args.out_dir),
    ]
    print(f'friendster: {n:,} nodes, {e:,} undirected edges -> <= {m:,} CSC entries')
    for name, b, where in rows:
        print(f'  {name:<38} {b / _GB:6.1f} GB  ({where})')
    final = sum(b for name, b, _ in rows[1:])
    peak_csc = rows[0][1] + rows[1][1] + rows[2][1]
    free = shutil.disk_usage(args.out_dir if args.out_dir.exists() else args.out_dir.parent).free
    print(f'  final dataset                          {final / _GB:6.1f} GB')
    print(f'  peak during csc (tmp + graph)          {peak_csc / _GB:6.1f} GB')
    print(f'  peak overall if feat runs after csc    {max(peak_csc, final) / _GB:6.1f} GB')
    print(f'  free now                               {free / _GB:6.1f} GB')
    print(f'  RAM peak (csc stage)                   ~{(m * 8 + 6 * _GB) / _GB:.0f} GB')


# ---------------------------------------------------------------------- stage: edges

_HEADER_RE = re.compile(rb'Nodes:\s*(\d+)\s+Edges:\s*(\d+)')


def _parse_block(buf):
    """Whitespace-separated id pairs -> (src int32, dst int32, lines, max, min).

    np.fromstring stops silently at the first token it cannot parse, so the pair
    count is checked against the newline count -- a short parse is an error, not a
    smaller graph.
    """
    a = np.fromstring(buf, dtype=np.int64, sep=' ')
    lines = buf.count(b'\n')
    if a.size != 2 * lines:
        raise ValueError(f'parsed {a.size} ids from {lines} lines -- malformed block')
    if a.size == 0:
        return np.empty(0, np.int32), np.empty(0, np.int32), 0, -1, 0
    hi, lo = int(a.max()), int(a.min())
    if lo < 0 or hi >= 2 ** 31:
        raise ValueError(f'SNAP id out of int32 range: [{lo}, {hi}]')
    return a[0::2].astype(np.int32), a[1::2].astype(np.int32), lines, hi, lo


def _text_blocks(gz_path, block_bytes, header):
    """Yield newline-aligned text blocks from `pigz -dc`, header lines stripped."""
    tool = 'pigz' if shutil.which('pigz') else 'gzip'
    proc = subprocess.Popen([tool, '-dc', str(gz_path)], stdout=subprocess.PIPE,
                            bufsize=block_bytes)
    carry = b''
    first = True
    try:
        while True:
            chunk = proc.stdout.read(block_bytes)
            if not chunk:
                break
            buf = carry + chunk
            cut = buf.rfind(b'\n') + 1
            carry, buf = buf[cut:], buf[:cut]
            if first:
                first = False
                pos = 0
                while buf.startswith(b'#', pos):
                    end = buf.index(b'\n', pos) + 1
                    m = _HEADER_RE.search(buf, pos, end)
                    if m:
                        header['nodes'], header['edges'] = int(m.group(1)), int(m.group(2))
                    pos = end
                buf = buf[pos:]
            if buf:
                yield buf
        if carry.strip():
            yield carry + b'\n'
        if proc.wait() != 0:
            raise RuntimeError(f'{tool} -dc {gz_path} exited with {proc.returncode}')
    finally:
        # On an early exit pigz just dies of SIGPIPE; don't mask the real error.
        proc.stdout.close()
        proc.wait()


def stage_edges(args):
    tmp = args.tmp_dir
    _need_disk(tmp, 2 * _SNAP_EDGES * 4, 'edges (tmp int32 src/dst)')
    src_path, dst_path = tmp / 'src.i32', tmp / 'dst.i32'
    header = {}
    t0 = time.time()

    # Bounded window: ProcessPoolExecutor.map / Pool.imap drain the whole generator up
    # front, which would pull all ~31 GB of text into RAM. Keep at most `window` blocks
    # in flight and drain in submission order, so the edge order on disk is the file's.
    window = args.workers + 2
    edges = lines_total = 0
    max_id = -1
    with ProcessPoolExecutor(args.workers) as pool, \
            open(str(src_path) + '.tmp', 'wb') as fs, open(str(dst_path) + '.tmp', 'wb') as fd:
        inflight = deque()
        bar = tqdm(desc='parse', unit='edge', unit_scale=True)

        def drain_one():
            nonlocal edges, lines_total, max_id
            s, d, lines, hi, _ = inflight.popleft().result()
            s.tofile(fs)
            d.tofile(fd)
            edges += s.size
            lines_total += lines
            max_id = max(max_id, hi)
            bar.update(s.size)

        for block in _text_blocks(args.raw, args.block_mb << 20, header):
            inflight.append(pool.submit(_parse_block, block))
            if len(inflight) >= window:
                drain_one()
        while inflight:
            drain_one()
        bar.close()
    print(f'parsed {edges:,} edges, max SNAP id {max_id:,} ({time.time() - t0:.0f}s)', flush=True)
    if header and header.get('edges') != edges:
        raise RuntimeError(f"header says {header['edges']:,} edges, parsed {edges:,}")
    os.replace(str(src_path) + '.tmp', src_path)
    os.replace(str(dst_path) + '.tmp', dst_path)

    # Compact the sparse SNAP ids to 0..N-1, keeping their relative order.
    print('compacting SNAP ids...', flush=True)
    src = np.memmap(src_path, mode='r+', dtype=np.int32, shape=(edges,))
    dst = np.memmap(dst_path, mode='r+', dtype=np.int32, shape=(edges,))
    present = np.zeros(max_id + 1, dtype=bool)
    step = args.chunk_edges
    for s in tqdm(range(0, edges, step), desc='presence', unit='chunk'):
        present[src[s:s + step]] = True
        present[dst[s:s + step]] = True
    num_nodes = int(present.sum())
    if header and header.get('nodes') != num_nodes:
        raise RuntimeError(f"header says {header['nodes']:,} nodes, edges touch {num_nodes:,}")
    old_to_new = (np.cumsum(present, dtype=np.int64) - 1).astype(np.int32)
    for s in tqdm(range(0, edges, step), desc='remap', unit='chunk'):
        src[s:s + step] = old_to_new[src[s:s + step]]
        dst[s:s + step] = old_to_new[dst[s:s + step]]
    src.flush()
    dst.flush()
    del src, dst, old_to_new

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / 'snap_ids.npy', np.flatnonzero(present).astype(np.int32))
    _write_json(_edges_meta_path(tmp), {
        'num_nodes': num_nodes, 'num_undirected_edges': int(edges),
        'max_snap_id': int(max_id), 'header': header, 'source': str(args.raw),
    })
    print(f'edges stage done: {num_nodes:,} nodes, {edges:,} edges '
          f'({time.time() - t0:.0f}s)', flush=True)


# ------------------------------------------------------------------------ stage: csc

def stage_csc(args):
    tmp, out = args.tmp_dir, args.out_dir
    meta_path = _edges_meta_path(tmp)
    if not meta_path.is_file():
        raise SystemExit(f'{meta_path} missing -- run --stage edges first')
    meta = json.loads(meta_path.read_text())
    n, e = int(meta['num_nodes']), int(meta['num_undirected_edges'])
    cap = 2 * e + n
    _need_disk(out, cap * 8 + (n + 1) * 8 + n * 4, 'csc (indices + indptr + nc_score)')
    src = np.memmap(tmp / 'src.i32', mode='r', dtype=np.int32, shape=(e,))
    dst = np.memmap(tmp / 'dst.i32', mode='r', dtype=np.int32, shape=(e,))
    step = args.chunk_edges
    t0 = time.time()

    # Raw column degree (both directions) -> contiguous column ranges ("buckets") of
    # ~bucket_entries each. A column's entries all land in one bucket, so sorting the
    # bucket's (col * N + row) keys sorts every column in it, and adjacent-equal keys
    # are exactly the duplicate entries to drop.
    deg = np.zeros(n, dtype=np.int64)
    for s in tqdm(range(0, e, step), desc='degree', unit='chunk'):
        deg += np.bincount(src[s:s + step], minlength=n)
        deg += np.bincount(dst[s:s + step], minlength=n)
    per_col = deg + 1                      # + the self-loop slot
    cum = np.cumsum(per_col)
    nb = max(1, int(np.ceil(cum[-1] / args.bucket_entries)))
    bounds = np.searchsorted(cum, np.arange(1, nb) * (cum[-1] / nb)).astype(np.int64)
    bounds = np.unique(np.r_[0, bounds, n])
    nb = bounds.size - 1
    sizes = np.add.reduceat(per_col, bounds[:-1])
    del deg, cum
    print(f'{nb} buckets, {int(sizes.sum()):,} raw entries incl. self-loops '
          f'({int(sizes.sum()) * 8 / _GB:.1f} GB resident)', flush=True)

    nn = np.int64(n)
    buckets, fill = [], np.zeros(nb, dtype=np.int64)
    for b in range(nb):
        lo, hi = int(bounds[b]), int(bounds[b + 1])
        arr = np.empty(int(sizes[b]), dtype=np.int64)
        r = np.arange(lo, hi, dtype=np.int64)
        arr[:hi - lo] = r * nn + r        # self-loops first; the sort puts them in place
        fill[b] = hi - lo
        buckets.append(arr)

    inner = bounds[1:-1]
    for s in tqdm(range(0, e, step), desc='scatter', unit='chunk'):
        a = src[s:s + step].astype(np.int64)
        c = dst[s:s + step].astype(np.int64)
        for col, row in ((c, a), (a, c)):  # edge (a, c) goes into column c and column a
            keys = col * nn + row
            bid = np.searchsorted(inner, col, side='right').astype(np.uint16)
            order = np.argsort(bid, kind='stable')     # radix sort on uint16
            keys = keys[order]
            counts = np.bincount(bid, minlength=nb)
            off = 0
            for b in np.flatnonzero(counts):
                k = int(counts[b])
                buckets[b][fill[b]:fill[b] + k] = keys[off:off + k]
                fill[b] += k
                off += k
        del a, c, keys, bid, order
    if not np.array_equal(fill, sizes):
        raise RuntimeError('bucket fill does not match the degree count')
    del src, dst

    new_deg = np.zeros(n, dtype=np.int64)
    ref_count = np.zeros(n, dtype=np.int64)
    written = 0
    idx_tmp = out / 'indices.dat.tmp'
    with open(idx_tmp, 'wb') as f:
        for b in tqdm(range(nb), desc='sort+dedup', unit='bucket'):
            lo, hi = int(bounds[b]), int(bounds[b + 1])
            k = buckets[b]
            buckets[b] = None              # release as we go
            k.sort()
            keep = np.empty(k.size, dtype=bool)
            keep[0] = True
            np.not_equal(k[1:], k[:-1], out=keep[1:])
            k = k[keep]
            col = k // nn
            row = k - col * nn
            del k, keep
            if int(np.count_nonzero(row == col)) != hi - lo:
                raise RuntimeError(f'bucket {b}: self-loop count != {hi - lo}')
            new_deg[lo:hi] = np.bincount(col - lo, minlength=hi - lo)
            ref_count += np.bincount(row, minlength=n)
            row.tofile(f)
            written += row.size
            del col, row

    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(new_deg, out=indptr[1:])
    if int(indptr[-1]) != written:
        raise RuntimeError(f'indptr total {int(indptr[-1])} != written {written}')
    indptr.tofile(str(out / 'indptr.dat') + '.tmp')
    os.replace(idx_tmp, out / 'indices.dat')
    os.replace(str(out / 'indptr.dat') + '.tmp', out / 'indptr.dat')

    # Same formula prepare_dataset_sym.py uses for Ginex's neighbor cache. On a
    # symmetric graph it is 1 everywhere; FALCON never reads it.
    score = ref_count.astype(np.float32) / (new_deg.astype(np.float32) + np.float32(1e-8))
    torch.save(torch.from_numpy(score), str(out / 'nc_score.pth') + '.tmp')
    os.replace(str(out / 'nc_score.pth') + '.tmp', out / 'nc_score.pth')

    _update_conf(
        out,
        num_nodes=n, num_edges=written,
        indptr_shape=[n + 1], indptr_dtype='int64',
        indices_shape=[written], indices_dtype='int64',
        csc_layout='symmetric', self_loop=True, nc_score='ginex',
        raw_undirected_edges=e, source='SNAP com-friendster.ungraph.txt.gz',
        node_ids='snap_ids.npy[new_id] = SNAP id (ascending SNAP order)',
    )
    dup = 2 * e + n - written
    print(f'csc done: {written:,} entries = 2x{e:,} + {n:,} self-loops - {dup:,} duplicates; '
          f'degree min/max {int(new_deg.min())}/{int(new_deg.max())} '
          f'({time.time() - t0:.0f}s)', flush=True)

    if args.keep_tmp:
        print(f'keeping tmp edges in {tmp}', flush=True)
    else:
        for p in ('src.i32', 'dst.i32', 'edges_meta.json'):
            (tmp / p).unlink(missing_ok=True)
        try:
            tmp.rmdir()
        except OSError:
            pass
        print(f'removed tmp edges from {tmp}', flush=True)


# ----------------------------------------------------------------------- stage: feat

def _num_nodes(out_dir):
    conf_path = out_dir / 'conf.json'
    if conf_path.is_file() and 'num_nodes' in json.loads(conf_path.read_text()):
        return int(json.loads(conf_path.read_text())['num_nodes'])
    if (out_dir / 'snap_ids.npy').is_file():
        return int(np.load(out_dir / 'snap_ids.npy', mmap_mode='r').shape[0])
    raise SystemExit('node count unknown -- run --stage edges (or csc) first')


def stage_feat(args):
    out = args.out_dir
    n, d, c = _num_nodes(out), args.feat_dim, args.num_classes
    _need_disk(out, n * d * 4 + n * 4, 'feat (features.dat + labels.dat)')
    t0 = time.time()

    feat_tmp = str(out / 'features.dat') + '.tmp'
    feats = np.memmap(feat_tmp, mode='w+', dtype=np.float32, shape=(n, d))
    rng = np.random.default_rng(args.seed)
    for s in tqdm(range(0, n, _FEAT_CHUNK_ROWS), desc='features', unit='chunk'):
        rows = min(_FEAT_CHUNK_ROWS, n - s)
        feats[s:s + rows] = rng.random((rows, d), dtype=np.float32)
    feats.flush()
    del feats
    os.replace(feat_tmp, out / 'features.dat')

    labels = np.random.default_rng(args.seed + 1).integers(0, c, size=n).astype(np.float32)
    labels.tofile(str(out / 'labels.dat') + '.tmp')
    os.replace(str(out / 'labels.dat') + '.tmp', out / 'labels.dat')
    del labels

    perm = np.random.default_rng(args.seed + 2).permutation(n)
    n_tr, n_va, n_te = (int(round(n * f)) for f in (args.train_frac, args.valid_frac, args.test_frac))
    if n_tr + n_va + n_te > n:
        raise SystemExit('split fractions sum to more than 1')
    parts = np.split(perm[:n_tr + n_va + n_te], [n_tr, n_tr + n_va])
    split = {k: torch.from_numpy(np.sort(p).astype(np.int64))
             for k, p in zip(('train', 'valid', 'test'), parts)}
    del perm
    torch.save(split, str(out / 'split_idx.pth') + '.tmp')
    os.replace(str(out / 'split_idx.pth') + '.tmp', out / 'split_idx.pth')

    _update_conf(
        out,
        num_nodes=n,
        features_shape=[n, d], features_dtype='float32',
        labels_shape=[n], labels_dtype='float32', num_classes=c,
        features_source=f'random uniform [0,1), seed={args.seed}',
        labels_source=f'random uniform over {c} classes, seed={args.seed + 1} '
                      '(chance-level accuracy by construction)',
        split_source=f'random permutation, seed={args.seed + 2}, fractions '
                     f'{args.train_frac}/{args.valid_frac}/{args.test_frac}',
    )
    print(f"feat done: features {n:,}x{d}, {c} classes, split "
          f"train={n_tr:,} valid={n_va:,} test={n_te:,} ({time.time() - t0:.0f}s)", flush=True)


# --------------------------------------------------------------------- stage: verify

def stage_verify(args):
    out = args.out_dir
    conf = json.loads((out / 'conf.json').read_text())
    need = ['num_nodes', 'indptr_shape', 'indices_shape', 'features_shape', 'labels_shape',
            'num_classes', 'csc_layout', 'self_loop']
    missing = [k for k in need if k not in conf]
    if missing:
        raise SystemExit(f'conf.json lacks {missing} -- a stage has not run yet')
    n, m = int(conf['num_nodes']), int(conf['indices_shape'][0])
    d = int(conf['features_shape'][1])
    sizes = {'indptr.dat': (n + 1) * 8, 'indices.dat': m * 8,
             'features.dat': n * d * 4, 'labels.dat': n * 4}
    for name, want in sizes.items():
        got = (out / name).stat().st_size
        if got != want:
            raise RuntimeError(f'{name}: {got} bytes, expected {want}')
    indptr = np.fromfile(out / 'indptr.dat', dtype=np.int64)
    indices = np.memmap(out / 'indices.dat', mode='r', dtype=np.int64, shape=(m,))
    if indptr[0] != 0 or int(indptr[-1]) != m or np.any(np.diff(indptr) < 1):
        raise RuntimeError('indptr is not a valid CSC offset array with >=1 entry per column')

    for s in tqdm(range(0, m, 1 << 28), desc='index range', unit='chunk'):
        seg = indices[s:s + (1 << 28)]
        if int(seg.min()) < 0 or int(seg.max()) >= n:
            raise RuntimeError('neighbor id out of range')

    labels = np.fromfile(out / 'labels.dat', dtype=np.float32)
    if labels.min() < 0 or labels.max() >= conf['num_classes'] or np.any(labels != np.round(labels)):
        raise RuntimeError('labels are not integers in [0, num_classes)')
    split = torch.load(out / 'split_idx.pth', weights_only=True)
    seen = np.zeros(n, dtype=bool)
    for k in ('train', 'valid', 'test'):
        ids = split[k].numpy()
        if ids.size and (ids.min() < 0 or ids.max() >= n):
            raise RuntimeError(f'split {k} out of range')
        if seen[ids].any():
            raise RuntimeError(f'split {k} overlaps another split')
        seen[ids] = True

    # Sampled structural checks: column sorted + unique, self-loop present, and for a
    # sampled entry (row v in column c) the reverse entry (row c in column v) exists.
    rng = np.random.default_rng(0)
    cols = rng.integers(0, n, size=args.verify_samples)
    bad_sorted = bad_self = bad_sym = 0
    for c in tqdm(cols, desc='symmetry', unit='col'):
        c = int(c)
        seg = np.asarray(indices[indptr[c]:indptr[c + 1]])
        if seg.size > 1 and not np.all(seg[1:] > seg[:-1]):
            bad_sorted += 1
        j = int(np.searchsorted(seg, c))
        if j >= seg.size or seg[j] != c:
            bad_self += 1
        v = int(seg[rng.integers(0, seg.size)])
        rev = np.asarray(indices[indptr[v]:indptr[v + 1]])
        j = int(np.searchsorted(rev, c))
        if j >= rev.size or rev[j] != c:
            bad_sym += 1
    print(f'verify: {n:,} nodes, {m:,} entries, avg degree {m / n:.1f}; sampled '
          f'{args.verify_samples:,} columns -> unsorted={bad_sorted} no-self-loop={bad_self} '
          f'asymmetric={bad_sym}', flush=True)
    if bad_sorted or bad_self or bad_sym:
        raise RuntimeError('structural check failed')
    print('verify OK', flush=True)


# ---------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', choices=['plan', 'edges', 'csc', 'feat', 'verify', 'all'],
                    default='plan')
    ap.add_argument('--raw', type=Path, default=DEFAULT_RAW)
    ap.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--tmp-dir', type=Path, default=DEFAULT_TMP)
    ap.add_argument('--keep-tmp', action='store_true',
                    help='keep the int32 edge files after csc (14.4 GB)')
    ap.add_argument('--workers', type=int, default=8, help='parse processes (edges stage)')
    ap.add_argument('--block-mb', type=int, default=128, help='text block size per parse task')
    ap.add_argument('--chunk-edges', type=int, default=50_000_000)
    ap.add_argument('--bucket-entries', type=int, default=200_000_000,
                    help='CSC entries sorted per bucket in csc; ~24 B RAM transient each')
    ap.add_argument('--feat-dim', type=int, default=128)
    ap.add_argument('--num-classes', type=int, default=20)
    ap.add_argument('--train-frac', type=float, default=0.01)
    ap.add_argument('--valid-frac', type=float, default=0.001)
    ap.add_argument('--test-frac', type=float, default=0.002)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--verify-samples', type=int, default=100_000)
    args = ap.parse_args()
    args.raw, args.out_dir, args.tmp_dir = (p.resolve() for p in (args.raw, args.out_dir, args.tmp_dir))

    if args.stage == 'plan':
        stage_plan(args)
        return
    if not args.raw.is_file() and args.stage in ('edges', 'all'):
        raise FileNotFoundError(args.raw)
    stages = ['edges', 'csc', 'feat', 'verify'] if args.stage == 'all' else [args.stage]
    for st in stages:
        print(f'\n===== stage {st} =====', flush=True)
        globals()[f'stage_{st}'](args)

    if args.stage == 'all':
        rel = os.path.relpath(args.out_dir, _REPO)
        print('\nnext (from the repo root):\n'
              f'  python preprocess/do_metis_new.py --dataset friendster --metis-k 1000 '
              f'--csc-dir {rel} --validate-symmetry\n'
              '  python -m sampling.intra_edges --dataset friendster --part-k 1000   # optional',
              flush=True)


if __name__ == '__main__':
    main()
