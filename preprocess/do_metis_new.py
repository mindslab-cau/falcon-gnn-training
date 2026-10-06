"""Full-graph METIS partitioning -- no chunking.

Replaces do_metis.py's chunked subgraph approach. With chunking
(do_metis.py with --metis-chunk-size 1_000_000), 98%+ of edges crossed
chunk boundaries on papers100M, so METIS partitioned 112 disconnected
node-id ranges. The result was perfectly balanced but had no community
structure (modularity Q ~ 0.016, edge cut ratio 0.984).

This script feeds the entire graph to METIS in one call via pymetis's
direct CSR ingestion (no .metis text dump on disk).

Resource profile (papers100M, 111M nodes / 3.2B directed CSC entries):
  - Peak RAM: ~60-100 GB (METIS coarsening hierarchy is the dominant term).
    Leaves ~20-60 GB headroom on a 128 GB box; keep other RAM users idle
    during the run.
  - Peak disk: <2 GB (only the output file). pymetis passes the CSR arrays
    straight to METIS in memory, so we skip the gpmetis text-format step.
  - Time: 1-3 hours for k <= 32; longer for larger k.

Outputs (matches do_metis.py contract):
  - {output_dir}/part_id.pth
  - {output_dir}/conf.json    (snapshot of dataset conf + METIS keys)

This script does NOT modify do_metis.py. By default it also does NOT
modify the dataset's conf.json -- pass --update-dataset-conf to do so.

Symmetry assumption:
  METIS treats the input as an undirected graph, so the CSC must be symmetric
  (each undirected edge stored as both (u,v) and (v,u)). papers100M as
  preprocessed by offgs is symmetrized; --validate-symmetry can verify this
  cheaply (samples 1M directed edges and checks reverse existence).

Dependencies:
  pip install pymetis

Usage (from the repo root). --dataset names the output folder, which
sampling.intra_edges._dataset_paths looks up as cluster/<name>-k<k>/metis/part_id.pth,
so use exactly these names:
  python preprocess/do_metis_new.py --dataset ogbn-products    --metis-k 1000 --csc-dir data/ogbn-products
  python preprocess/do_metis_new.py --dataset ogbn-papers100M  --metis-k 1000 --csc-dir data/ogbn-papers100M-sym
  python preprocess/do_metis_new.py --dataset friendster       --metis-k 1000 --csc-dir data/friendster/sym
"""
import argparse
import gc
import json
import os
import time

import numpy as np
import torch

# ── OOM 디버깅용 메모리 리포터 ────────────────────────────────────────────────
_MEM_T0 = time.time()


def _mem(label=""):
    """Print current process RSS + system available RAM. Used to pinpoint the
    step at which the OOM killer fires (the last printed label before the
    process dies is the culprit). No external deps (reads /proc)."""
    rss_gb = avail_gb = total_gb = float("nan")
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_gb = int(line.split()[1]) / 1024 / 1024  # kB -> GB
                    break
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                k, v = line.split(":", 1)
                info[k] = int(v.split()[0])  # kB
            avail_gb = info.get("MemAvailable", 0) / 1024 / 1024
            total_gb = info.get("MemTotal", 0) / 1024 / 1024
    except Exception:
        pass
    dt = time.time() - _MEM_T0
    print(
        f"[MEM +{dt:6.1f}s] RSS={rss_gb:6.1f}GB  "
        f"sys_avail={avail_gb:6.1f}/{total_gb:.1f}GB  <- {label}",
        flush=True,
    )


def _load_csc(graph_path):
    print(f"Loading graph from {graph_path}")
    t0 = time.time()
    g = torch.load(graph_path, weights_only=False)
    if not (hasattr(g, "csc_indptr") and hasattr(g, "indices")):
        raise ValueError("Expected GraphBolt CSC graph (csc_indptr + indices).")
    csc_indptr = g.csc_indptr.to(torch.int64).cpu()
    indices = g.indices.to(torch.int32).cpu()
    print(
        f"  csc_indptr: {tuple(csc_indptr.shape)} int64, "
        f"indices: {tuple(indices.shape)} int32  "
        f"({time.time() - t0:.1f}s)"
    )
    del g
    gc.collect()
    return csc_indptr, indices


def _load_csc_dir(csc_dir):
    conf_path = os.path.join(csc_dir, "conf.json")
    print(f"Loading Ginex CSC from {csc_dir}")
    t0 = time.time()
    with open(conf_path) as f:
        conf = json.load(f)
    indptr_np = np.fromfile(
        os.path.join(csc_dir, "indptr.dat"), dtype=conf["indptr_dtype"]
    )
    indices_np = np.memmap(
        os.path.join(csc_dir, "indices.dat"), mode="r",
        dtype=conf["indices_dtype"], shape=tuple(conf["indices_shape"]),
    )
    if indptr_np.shape[0] != int(conf["num_nodes"]) + 1:
        raise ValueError("indptr length does not match num_nodes")
    if int(indptr_np[-1]) != indices_np.shape[0]:
        raise ValueError("indptr[-1] does not match indices length")
    csc_indptr = torch.from_numpy(indptr_np).to(torch.int64)
    indices = torch.from_numpy(indices_np).to(torch.int32)
    del indptr_np, indices_np
    gc.collect()
    print(
        f"  csc_indptr: {tuple(csc_indptr.shape)} int64, "
        f"indices: {tuple(indices.shape)} int32  "
        f"({time.time() - t0:.1f}s)"
    )
    return csc_indptr, indices


def _drop_self_loops(csc_indptr, indices):
    """METIS rejects self-loops. Filter them out and rebuild indptr."""
    print("Filtering self-loops (METIS requirement)...")
    t0 = time.time()
    N = int(csc_indptr.numel() - 1)
    counts = (csc_indptr[1:] - csc_indptr[:-1]).to(torch.int32)
    columns = torch.repeat_interleave(
        torch.arange(N, dtype=torch.int32), counts
    )
    keep = indices != columns
    n_self = int((~keep).sum().item())
    if n_self == 0:
        print(f"  no self-loops found  ({time.time() - t0:.1f}s)")
        del columns, keep, counts
        gc.collect()
        return csc_indptr, indices

    print(f"  found {n_self:,} self-loops, rebuilding CSC...")
    new_indices = indices[keep].clone()
    keep_long = keep.to(torch.int64)
    keep_per_col = torch.zeros(N, dtype=torch.int64)
    keep_per_col.scatter_add_(0, columns.to(torch.int64), keep_long)
    new_indptr = torch.zeros(N + 1, dtype=torch.int64)
    new_indptr[1:] = torch.cumsum(keep_per_col, dim=0)

    del columns, keep, keep_long, counts, keep_per_col
    gc.collect()
    print(f"  rebuilt in {time.time() - t0:.1f}s")
    return new_indptr, new_indices


def _validate_symmetry(csc_indptr, indices, sample_size=1_000_000, seed=0):
    """Spot-check that the CSC is symmetric. METIS assumes undirected input."""
    print(f"Symmetry check on {sample_size:,} sampled directed entries...")
    t0 = time.time()
    rng = np.random.default_rng(seed)
    E = int(indices.numel())
    if sample_size > E:
        sample_size = E
    sample_idx = rng.choice(E, size=sample_size, replace=False)

    # For each sampled directed entry i in CSC: src = indices[i], dst = column(i).
    # We need to check that there's a directed entry in indices[csc_indptr[src]:csc_indptr[src+1]]
    # equal to dst (i.e. (dst, src) also exists -> symmetric).
    sample_idx_t = torch.from_numpy(sample_idx).to(torch.int64)
    src = indices[sample_idx_t].to(torch.int64)
    # find dst (column) for each sampled entry via searchsorted on indptr
    dst = torch.searchsorted(csc_indptr, sample_idx_t, right=True) - 1

    asymmetric = 0
    for k in range(sample_size):
        s = int(src[k].item())
        d = int(dst[k].item())
        ns = indices[csc_indptr[s].item():csc_indptr[s + 1].item()]
        if not (ns == d).any():
            asymmetric += 1
    rate = asymmetric / sample_size
    print(f"  asymmetric sampled entries: {asymmetric}/{sample_size}  "
          f"(rate={rate:.4%}, took {time.time() - t0:.1f}s)")
    if rate > 0.01:
        raise RuntimeError(
            "Graph appears asymmetric (>1% of sampled edges have no reverse). "
            "METIS expects undirected/symmetric input. Symmetrize first."
        )


def _run_pymetis(xadj_t, adjncy_t, K, recursive, verbose=False):
    """Call pymetis on already-int64 torch tensors.

    Caller must have pre-converted adjncy to int64 and freed the original
    int32 tensor before this call. That extra discipline is what saves us
    from OOM on papers100M: pymetis with int32 input internally allocates
    its own int64 copy of adjncy (12.8 GB on papers100M), which lives next
    to our int32 throughout the METIS run. Pre-converting in the caller
    sidesteps that double allocation.

    On top of that, this function uses pymetis's modern API:
      - CSRAdjacency (zero-copy adjacency object when dtypes already match)
      - Options(ctype=RM, niter=3, ncuts=1) -- the lightest coarsening +
        minimal refinement, trading a few % cut quality for several GB of
        METIS internal working memory.

    Mode notes (see prior version of this docstring for the full breakdown):
      recursive=True  -> METIS_PartGraphRecursive, lower peak RAM, recommended
      recursive=False -> METIS_PartGraphKway, OOM's on papers100M with 128GB
    """
    import pymetis

    mode = "recursive bisection" if recursive else "k-way"
    print(f"Running METIS (k={K}, mode={mode}) -- this is the peak-RAM step.")
    print("  pymetis -> METIS C library; no temp files on disk.")

    # Accept torch tensors OR numpy arrays (incl. np.memmap for --mmap-input).
    xadj = xadj_t if isinstance(xadj_t, np.ndarray) else xadj_t.numpy()
    adjncy = adjncy_t if isinstance(adjncy_t, np.ndarray) else adjncy_t.numpy()
    print(
        f"  xadj({xadj.dtype}, {xadj.shape}) + "
        f"adjncy({adjncy.dtype}, {adjncy.shape})  "
        f"-> {(xadj.nbytes + adjncy.nbytes) / 1e9:.1f} GB CSR resident"
    )

    _mem("_run_pymetis: after .numpy() views (xadj/adjncy)")
    # Zero-copy adjacency. In pymetis 2025.x the CSRAdjacency ctor takes
    # (adj_starts, adjacent) -- NOT (xadj, adjncy). Zero-copy only happens when
    # both arrays already match pymetis.zero_copy_dtype() (int64 in this build);
    # otherwise pymetis copies. We pre-convert to int64 in run(), so this is a
    # true zero-copy and avoids a second ~26 GB int64 copy of adjncy.
    csr = None
    if hasattr(pymetis, "CSRAdjacency"):
        zc = getattr(pymetis, "zero_copy_dtype", lambda: None)()
        if zc is not None and (xadj.dtype != zc or adjncy.dtype != zc):
            print(f"  WARNING: arrays are {xadj.dtype}/{adjncy.dtype} but zero-copy "
                  f"dtype is {zc}; CSRAdjacency will COPY (extra RAM).")
        try:
            csr = pymetis.CSRAdjacency(adj_starts=xadj, adjacent=adjncy)
            print("  using pymetis.CSRAdjacency(adj_starts, adjacent) -- zero-copy")
        except Exception as e:
            print(f"  CSRAdjacency creation failed: {e!r}; using xadj/adjncy fallback")
    _mem("_run_pymetis: after CSRAdjacency (should add ~0 if zero-copy)")

    # Memory-saving options. This pymetis version configures options via
    # _set(OptionKey.X, value) + enum values -- setattr(options, "ctype", ...)
    # does NOT reach the C library. We set the lightest coarsening (RM) and
    # trim refinement/cuts, then read the values back to PROVE they took.
    options = None
    if hasattr(pymetis, "Options") and hasattr(pymetis, "OptionKey"):
        try:
            options = pymetis.Options()
            options.set_defaults()
            OK = pymetis.OptionKey
            wanted = []  # (name, OptionKey, value)
            if hasattr(pymetis, "CType") and hasattr(pymetis.CType, "RM"):
                wanted.append(("ctype", OK.CTYPE, int(pymetis.CType.RM)))  # RM < SHEM RAM
            wanted += [
                ("niter", OK.NITER, 3),    # default 10; fewer refinement passes
                ("ncuts", OK.NCUTS, 1),    # keep a single partitioning in memory
            ]
            if hasattr(OK, "NO2HOP"):
                wanted.append(("no2hop", OK.NO2HOP, 1))  # disable 2-hop matching
            if verbose and hasattr(OK, "DBGLVL") and hasattr(pymetis, "DebugLevel"):
                # METIS prints multilevel progress to stdout: coarsening levels
                # (nvtxs shrinking), each bisection, per-phase timing. This is the
                # only real "progress" signal -- part_graph is one opaque C call
                # tqdm cannot wrap. INFO|TIME|COARSEN keeps it informative but not
                # flooded (skip REFINE/MOVEINFO which log per-vertex).
                DL = pymetis.DebugLevel
                dbg = int(DL.INFO) | int(DL.TIME) | int(DL.COARSEN)
                wanted.append(("dbglvl", OK.DBGLVL, dbg))
            applied = []
            for name, key, val in wanted:
                options._set(key, val)
                got = options._get(key)
                applied.append(f"{name}={got}{'' if got == val else f'(!wanted {val})'}")
            print(f"  pymetis.Options applied (verified via _get): {', '.join(applied)}")
        except Exception as e:
            options = None
            print(f"  Options init failed: {e!r}; running with library defaults")

    kwargs = {"recursive": recursive}
    if options is not None:
        kwargs["options"] = options
    if csr is not None:
        kwargs["adjacency"] = csr
    else:
        kwargs["xadj"] = xadj
        kwargs["adjncy"] = adjncy

    _mem("_run_pymetis: RIGHT BEFORE part_graph (METIS C alloc starts now)")
    t0 = time.time()
    n_cuts, membership = pymetis.part_graph(K, **kwargs)
    elapsed = time.time() - t0
    print(f"  METIS done in {elapsed:.1f}s; reported edge cuts = {n_cuts:,}")
    _mem("_run_pymetis: after part_graph returned")

    del csr, options, xadj, adjncy
    gc.collect()
    _mem("_run_pymetis: after del csr/options/xadj/adjncy")
    return np.asarray(membership, dtype=np.int64)


def _write_summary_json(part_id, summary_path):
    """Per-cluster node counts, same {num_partitions, counts} contract as
    do_modularity.py's part_summary.json (data_clustering.py reads this in
    full-cluster mode)."""
    print(f"Computing per-cluster counts and writing {summary_path}")
    t0 = time.time()
    K = int(part_id.max().item()) + 1
    counts = torch.zeros(K, dtype=torch.int64)
    chunk = 10_000_000
    for s in range(0, part_id.numel(), chunk):
        e = min(s + chunk, part_id.numel())
        counts.add_(torch.bincount(part_id[s:e], minlength=K))
    nonempty = int((counts > 0).sum().item())
    counts_list = counts.tolist()
    del counts
    gc.collect()
    with open(summary_path, "w") as f:
        f.write("{\"num_partitions\": ")
        f.write(str(K))
        f.write(", \"counts\": {")
        first = True
        for cid, sz in enumerate(counts_list):
            if sz == 0:
                continue
            if not first:
                f.write(", ")
            f.write(f"\"{cid}\": {sz}")
            first = False
        f.write("}}")
    print(f"  K={K:,}, nonempty={nonempty:,}  ({time.time() - t0:.1f}s)")


def _write_conf_snapshot(source_conf_path, conf_out_path, args):
    conf = {}
    if source_conf_path and os.path.exists(source_conf_path):
        with open(source_conf_path, "r") as f:
            conf = json.load(f)
    conf["metis_k"] = int(args.metis_k)
    conf["metis_chunk_size"] = 0  # 0 == no chunking
    conf["metis_mode"] = "full_graph"
    conf["metis_partitioned"] = True
    conf["metis_library"] = "pymetis"
    with open(conf_out_path, "w") as f:
        json.dump(conf, f, indent=2)
    print(f"Wrote conf snapshot -> {conf_out_path}")


def run(args):
    print(f"=== METIS partitioning for {args.dataset} (k={args.metis_k}) ===")
    if args.metis_k <= 1:
        raise ValueError("--metis-k must be > 1.")
    print("=== Full-graph METIS partitioning ===")
    dataset_path = os.path.join(args.store_path, f"{args.dataset}-offgs")
    print("=== Dataset ===")
    graph_path = os.path.join(dataset_path, "graph.pth")
    source_conf_path = (
        os.path.join(args.csc_dir, "conf.json")
        if args.csc_dir else os.path.join(dataset_path, "conf.json")
    )

    # Default output: <repo>/cluster/<dataset>-k<k>/metis/ -- reuse the
    # existing per-(dataset,k) folder (e.g. ogbn-papers100M-k1000, which already
    # holds the modularity outputs) and keep metis results in a metis/ subfolder.
    # sampling.intra_edges._dataset_paths reads part_id.pth from there.
    # Override with --output-dir.
    print("=== Output ===")
    if args.output_dir:
        print(f"Using user-specified output dir: {args.output_dir}")
        out_dir = args.output_dir
    else:
        print("Using default output dir: <repo>/cluster/<dataset>-k<k>/metis/")
        cluster_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cluster"
        )
        print(f"  cluster_dir = {cluster_dir}")
        out_dir = os.path.join(
            cluster_dir, f"{args.dataset}-k{args.metis_k}", "metis"
        )
    os.makedirs(out_dir, exist_ok=True)
    print(f"  output_dir = {out_dir}")
    part_id_path = os.path.join(out_dir, "part_id.pth")
    summary_path = os.path.join(out_dir, "part_summary.json")
    conf_out_path = os.path.join(out_dir, "conf.json")
    print(f"  part_id.pth        -> {part_id_path}")
    if os.path.exists(part_id_path) and not args.force:
        raise RuntimeError(
            f"{part_id_path} already exists. Use --force to overwrite."
        )
    print(f"  part_summary.json  -> {summary_path}")
    _mem("start of run() (before load)")
    csc_indptr, indices = (
        _load_csc_dir(args.csc_dir) if args.csc_dir else _load_csc(graph_path)
    )
    print(f"  graph: shape={tuple(csc_indptr.shape)}, nnz={indices.numel()}")
    _mem("after _load_csc (csc_indptr int64 + indices int32 resident)")
    if args.validate_symmetry:
        _validate_symmetry(csc_indptr, indices, sample_size=args.symmetry_sample)
        _mem("after _validate_symmetry")
    print("Dropping self-loops (METIS requirement)...")
    csc_indptr, indices = _drop_self_loops(csc_indptr, indices)
    _mem("after _drop_self_loops")

    # Pre-convert adjncy to int64 BEFORE calling pymetis. METIS IDX is
    # 64-bit in this build, so passing int32 forces pymetis to allocate
    # its own int64 copy of the 12.8 GB adjncy array on top of ours --
    # that doubling is what pushed the previous run over 119 GB. Doing
    # the conversion here lets us drop the int32 tensor immediately, so
    # only the int64 version is alive when METIS starts.
    print("Pre-converting adjncy int32 -> int64 (saves ~12.8 GB during METIS)...")
    _mem("before int32->int64 (both int32 and int64 adjncy briefly coexist)")
    t_prep = time.time()
    adjncy_int64 = indices.to(torch.int64)
    del indices
    gc.collect()
    print(
        f"  done in {time.time() - t_prep:.1f}s; "
        f"adjncy now int64, {adjncy_int64.numel() * 8 / 1e9:.1f} GB resident"
    )
    _mem("after int32->int64 + del int32 (only int64 adjncy alive)")

    # Optionally push the input CSR to disk and hand METIS memory-mapped arrays.
    # These pages are file-backed (clean), so the kernel can evict them under
    # pressure instead of counting them against anonymous RAM -- freeing ~27 GB
    # of headroom for METIS's own coarsening allocations. The .npy files live in
    # mmap_dir and are deleted in the finally-block below (and by the watchdog on
    # a hard kill).
    xadj_arg, adjncy_arg = csc_indptr, adjncy_int64
    mmap_dir = None
    if args.mmap_input:
        mmap_dir = args.mmap_dir or os.path.join(out_dir, "_mmap_tmp")
        os.makedirs(mmap_dir, exist_ok=True)
        xadj_npy = os.path.join(mmap_dir, "xadj_i64.npy")
        adjncy_npy = os.path.join(mmap_dir, "adjncy_i64.npy")
        print(f"--mmap-input: dumping CSR to {mmap_dir} then memmapping...")
        t_mm = time.time()
        np.save(xadj_npy, csc_indptr.numpy())
        np.save(adjncy_npy, adjncy_int64.numpy())
        del csc_indptr, adjncy_int64
        gc.collect()
        xadj_arg = np.load(xadj_npy, mmap_mode="r")
        adjncy_arg = np.load(adjncy_npy, mmap_mode="r")
        print(f"  CSR now memmap-backed on disk ({time.time() - t_mm:.1f}s); "
              f"in-RAM CSR freed")
        _mem("after --mmap-input dump+memmap (CSR off anonymous RAM)")

    _mem("before _run_pymetis (PEAK-RAM step begins next)")
    try:
        membership = _run_pymetis(
            xadj_arg, adjncy_arg, args.metis_k, recursive=args.recursive,
            verbose=args.metis_verbose,
        )
    finally:
        if mmap_dir is not None:
            import shutil
            shutil.rmtree(mmap_dir, ignore_errors=True)
            print(f"  cleaned up mmap tmp dir {mmap_dir}")
    _mem("after _run_pymetis returned (membership allocated)")
    del xadj_arg, adjncy_arg
    gc.collect()
    _mem("after del csc_indptr/adjncy_int64")

    print(f"Writing part_id.pth -> {part_id_path}")
    part_id = torch.from_numpy(membership).to(torch.int64)
    torch.save(part_id, part_id_path)
    print(
        f"  part_id: shape={tuple(part_id.shape)}, "
        f"min={int(part_id.min().item())}, max={int(part_id.max().item())}"
    )
    del membership
    gc.collect()
    _mem("after writing part_id.pth")

    _write_summary_json(part_id, summary_path)
    del part_id
    gc.collect()
    _mem("after _write_summary_json")

    _write_conf_snapshot(source_conf_path, conf_out_path, args)

    if args.update_dataset_conf:
        _write_conf_snapshot(source_conf_path, source_conf_path, args)
        print(f"  (also updated dataset conf at {source_conf_path})")

    print("\n=== Done ===")
    print(f"  part_id.pth        -> {part_id_path}")
    print(f"  part_summary.json  -> {summary_path}")
    print(f"  conf.json          -> {conf_out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ogbn-papers100M")
    parser.add_argument(
        "--store-path", type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
        ),
    )
    parser.add_argument(
        "--csc-dir", type=str, default=None,
        help="Read Ginex-layout conf.json/indptr.dat/indices.dat directly instead of <store-path>/<dataset>-offgs/graph.pth.",
    )
    parser.add_argument(
        "--metis-k", type=int, required=True,
        help="number of partitions",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="use METIS recursive bisection (lower peak RAM, default for large graphs).",
    )
    parser.add_argument(
        "--no-recursive", dest="recursive", action="store_false",
        help="force k-way partitioning (higher peak RAM; only for small graphs).",
    )
    parser.set_defaults(recursive=True)
    parser.add_argument(
        "--validate-symmetry", action="store_true",
        help="spot-check that the CSC is symmetric (METIS requires undirected input).",
    )
    parser.add_argument(
        "--symmetry-sample", type=int, default=1_000_000,
        help="number of directed entries to sample for the symmetry check.",
    )
    parser.add_argument(
        "--metis-verbose", action="store_true",
        help="enable METIS DBGLVL (INFO|TIME|COARSEN): METIS prints multilevel "
             "progress -- coarsening levels, each bisection, per-phase timing -- "
             "to stdout. The only real progress signal, since part_graph is one "
             "opaque C call tqdm cannot wrap.",
    )
    parser.add_argument(
        "--mmap-input", action="store_true",
        help="dump the CSR arrays to disk and hand METIS memory-mapped (file-"
             "backed) arrays, so the kernel can evict them under RAM pressure "
             "instead of counting ~27GB against anonymous RAM. Trades disk for "
             "RAM headroom; pairs well with a swapfile for the OOM-prone k.",
    )
    parser.add_argument(
        "--mmap-dir", type=str, default=None,
        help="where --mmap-input writes its temp .npy files "
             "(default: {output_dir}/_mmap_tmp). Auto-deleted after the run.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Where part_id.pth and conf.json are written. "
             "Default: dataset directory (matches do_metis.py).",
    )
    parser.add_argument(
        "--update-dataset-conf", action="store_true",
        help="Also write the partition keys back into the dataset's conf.json.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(args)
    run(args)
