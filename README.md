# FALCON

Training code and data pipeline for FALCON: GNN training on large graphs (ogbn-products,
ogbn-papers100M, Friendster) that partitions the graph once with METIS and then trains on
cluster-local neighborhoods with per-epoch node subsampling.

Everything below is run from the repository root. Datasets, partitions and training
outputs are written under `data/`, `cluster/`, `main/output/` and `main/runs/`; all four
are git-ignored.

## 1. Requirements

| | Tested with |
|---|---|
| OS / Python | Linux, Python 3.11.14 |
| GPU stack | CUDA 12.8 (PyTorch cu128 build), NVIDIA driver 590 |
| Build tools | g++ 13.3 and ninja 1.13 (the C++ neighbor sampler in `sampling/cpp/` is JIT-compiled on first run) |
| Memory | METIS on papers100M needs 60-100 GB RAM; training keeps the graph in page cache |
| Disk | products ~10 GB, papers100M ~300 GB, Friendster ~200 GB (download + raw + graph + reordered copy) |

```bash
pip install -r requirements.txt
```

## 2. Prepare a dataset

Each dataset goes through four steps:

1. build the graph directory (`data/<graph>/`: `conf.json`, `indptr.dat`, `indices.dat`,
   `features.dat`, `labels.dat`, `split_idx.pth`),
2. partition it with METIS (`cluster/<name>-k1000/metis/part_id.pth`),
3. renumber the nodes by partition (`data/reordered/<dataset>-k1000/`), which is what
   training reads,
4. train.

The scripts do not download anything. For the OGB datasets, fetch the zip from OGB and
unzip it under `data/ogb/` first (step 0 below); `convert_ogb_raw.py` then reads the
extracted folder.

The `--dataset` value differs between tools: `do_metis_new.py` takes the *graph name*
(`ogbn-products`, `ogbn-papers100M`, `friendster`), while `reorder_metis.py` and
`run_falcon.py` take the short name (`products`, `papers`, `friendster`). The folder
names below are the ones the code looks up by default.

### 2.1 ogbn-products

```bash
# step 0: download (1.4 GB) and unzip -> data/ogb/products/
mkdir -p data/ogb && (cd data/ogb && wget http://snap.stanford.edu/ogb/data/nodeproppred/products.zip && unzip -q products.zip)

# step 1: data/ogb/products/ -> data/raw/products/*.bin -> data/ogbn-products/
python preprocess/convert_ogb_raw.py --dataset ogbn-products
python preprocess/prepare_dataset_sym.py --raw-dir data/raw/products --out-dir data/ogbn-products \
    --num-features 100 --num-classes 47

# step 2: METIS, k = 1000 partitions -> cluster/ogbn-products-k1000/metis/part_id.pth
python preprocess/do_metis_new.py --dataset ogbn-products --metis-k 1000 --csc-dir data/ogbn-products

# step 3: renumber -> data/reordered/products-k1000/
python preprocess/reorder_metis.py --dataset products --part-k 1000
```

### 2.2 ogbn-papers100M

`convert_ogb_raw.py` streams the arrays out of the extracted zip without loading them
into RAM. If the extracted `papers100M-bin/` folder lives elsewhere, pass it with
`--ogb-dir`.

```bash
# step 0: download (57 GB) and unzip -> data/ogb/papers100M-bin/
mkdir -p data/ogb && (cd data/ogb && wget http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip && unzip -q papers100M-bin.zip)

# step 1: data/ogb/papers100M-bin/ -> data/raw/papers100M/*.bin -> data/ogbn-papers100M-sym/   (symmetrized, self-loops added)
python preprocess/convert_ogb_raw.py --dataset ogbn-papers100M
python preprocess/prepare_dataset_sym.py            # defaults are the papers100M paths; needs ~26 GB in --tmp-dir (default /tmp)

# step 2 -> cluster/ogbn-papers100M-k1000/metis/part_id.pth   (1-3 h, 60-100 GB RAM)
python preprocess/do_metis_new.py --dataset ogbn-papers100M --metis-k 1000 --csc-dir data/ogbn-papers100M-sym

# step 3 -> data/reordered/papers-k1000/
python preprocess/reorder_metis.py --dataset papers --part-k 1000
```

### 2.3 Friendster

Download `com-friendster.ungraph.txt.gz` from SNAP
(https://snap.stanford.edu/data/com-Friendster.html) to
`data/friendster/raw/com-friendster.ungraph.txt.gz`. Friendster has no features or
labels; the script generates random ones deterministically, as DiskGNN does, so accuracy on it is
chance level by design and the dataset is for speed and memory measurements only.

```bash
# step 1 -> data/friendster/sym/        (`--stage plan` prints disk/RAM needs and writes nothing)
python preprocess/prepare_friendster.py --stage plan
python preprocess/prepare_friendster.py --stage all

# step 2 -> cluster/friendster-k1000/metis/part_id.pth
python preprocess/do_metis_new.py --dataset friendster --metis-k 1000 --csc-dir data/friendster/sym --validate-symmetry

# step 3 -> data/reordered/friendster-k1000/
python preprocess/reorder_metis.py --dataset friendster --part-k 1000
```

## 3. Train

`main/run_falcon.py` trains GraphSAGE or GCN (`--model sage|gcn`). The command below is
the full configuration used in the paper; change `--dataset`, `--data-dir` and
`--gpu-cache-gb` per run.

```bash
python main/run_falcon.py --dataset products --data-dir data/reordered/products-k1000 --part-k 1000 \
    --model sage --mode cluster --chunk 128 --epochs 30 \
    --graph rtintra --inter-keep-frac 0 --node-subsampling --factor 5 \
    --candidate-filter --seed-chunking --intra-cluster-shuffle --chunk-shuffle \
    --gpu-cache-gb 0.1 --pipeline --direct-pinned --late-gather --gil-release \
    --part-range --rej-nodiscard --tag my_run
```

| Option | Meaning |
|---|---|
| `--factor F` | node subsampling strength; each cluster keeps a `1/sqrt(F)` fraction of its non-training nodes per epoch (`--factor 1` = no subsampling) |
| `--inter-keep-frac p` | fraction of inter-cluster neighbors kept (0 = intra-cluster edges only) |
| `--gpu-cache-gb G` | size of the GPU feature cache; the paper used 0.1 for products and 2 for papers100M / Friendster |
| `--screen` | skip val/test evaluation and time training only |
| `--test-every N`, `--final-test K` | evaluate test every N epochs / re-evaluate the best-validation model K times at the end |

The first run compiles the two C++ samplers (`~/.cache/torch_extensions/`, 1-2 min).
Epoch 0 is excluded from timing as cache warm-up.

Outputs go to `main/output/<tag>/<model>/<timestamp>_.../`:
`meta.json`, `metrics.json`, `metrics.csv` (per-epoch loss, accuracy and timing),
`best_model.pt`, and training-curve PNGs when matplotlib is installed. A JSON record of
the run (configuration plus per-epoch history) is also written under
`main/runs/<tag>/<model>/`.

## 4. Repository layout

```
main/run_falcon.py            training script
model/models.py               GraphSAGE / GCN (PyTorch Geometric)
sampling/                     node subsampling, masked neighbor loaders, C++ samplers (sampling/cpp/)
preprocess/
  convert_ogb_raw.py          extracted OGB download -> flat .bin files
  prepare_dataset_sym.py      .bin files -> symmetric CSC graph directory
  prepare_friendster.py       SNAP Friendster -> CSC graph directory
  add_self_loops.py           add self-loops to an existing graph directory (not needed for the paths above)
  do_metis_new.py             METIS partitioning
  reorder_metis.py            renumber nodes by partition
requirements.txt
```
