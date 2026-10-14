"""FALCON sampling stage.

Intra-Cluster Subsampling: per-epoch stratified node subsampling over a fixed
partition, feeding the in-memory G* materialize pipeline.

Ported from our Ginex-based prototype but detached from Ginex's disk fanout sampler --
here the partition drives an in-memory graph, not a runtime C++ neighbor mask.

Stages:
    part.py             load/validate part_id (the METIS/Leiden partition)
    node_subsampler.py  per-epoch keep_mask + new_id + cluster_ptr  [Step B]
    intra_edges.py      Phase 0 precompute of symmetric intra-cluster edges [Step C]
    materialize.py      keep_mask + intra edges -> RAM-resident G*         [Step D]
"""
from .part import load_part_id, PartInfo
from .node_subsampler import ClusterNodeSubsampler, Subsample
from .intra_edges import build_intra_csc, load_intra_csc
from .materialize import materialize, MaterializedGraph

__all__ = [
    'load_part_id',
    'PartInfo',
    'ClusterNodeSubsampler',
    'Subsample',
    'build_intra_csc',
    'load_intra_csc',
    'materialize',
    'MaterializedGraph',
]
