"""FALCON 학습 스크립트 (v6).

결과는 main/output/<tag>/<model>/ 와 main/runs/<tag>/<model>/ 에 쌓인다.

옵션 요약 (--screen 이 없으면 매 epoch val/test 를 평가한다):
  --no-node-mask / --prep-ahead / --cache-fill degree
             offline intra 노드 마스크 제거, 다음 epoch 준비 중첩, 차수 기반 캐시 채움.
             캐시 구성 시 상세 진단과 epoch 준비 시간 상세 기록은 기본 제공.
  --screen   스크리닝 모드. val/test 를 건너뛰고 train_sec 만 잰다 (epoch 당 절반 시간).
             속도 후보를 빠르게 버리기 위한 모드이지 보고용이 아니다.
  --graph rtintra
             [S11] 오프라인 intra CSC 를 쓰지 않고, 원본 CSC 를 걸으면서 노드 마스크와
             intra 여부(part_id)를 샘플링 때 같이 판정한다. 같은 간선 집합을 다른 방법으로
             만드는 것이라 결과 분포는 --graph intra 와 같다.
               얻는 것: 오프라인 전처리(papers 26GB 스캔)와 그 산출물 8.2GB 가 사라지고,
                        학습/평가가 CSC 한 벌(원본)만 만진다.
               내는 것: 걷는 행이 1/0.611 = 1.64배 길어지고(intra 비율), col 이 int32 ->
                        int64 라 스캔 바이트가 2배, 이웃마다 part[c] 조회가 한 번 더 붙는다.
                        기각 샘플링 생존율도 0.611배가 되어 조회 k/p 와 폴백이 함께 는다.
             즉 '전처리와 메모리를 샘플링 시간으로 바꾸는' 거래다. 그 환율을 재는 것이 목적.
  --keep-inter-frac / --keep-inter-degree
             [S11-ki] keep-inter-too. train 노드 중 degree 하위 P 에 한해 inter-cluster
             간선을 살린다. rtintra 전용 -- 오프라인 intra CSC 는 inter 를 물리적으로 지운
             파일이라 '이 중심에서만 살린다'를 표현할 수 없다.
  --inter-keep-frac
             [R] inter 이웃 무작위 보존. 위 keep-inter 가 '어떤 노드를 통째로 봐줄까'라면
             이쪽은 '모든 노드에서 inter 이웃 중 몇 %를 봐줄까'다. 중심마다 자기 이웃에서
             intra 를 뺀 inter 쪽 각각을 확률 K 의 독립 동전으로 살린다.
             둘은 완전히 독립이고 같이 켜면 OR 로 합쳐진다. rtintra 전용.

             동전은 매 epoch 새로 던진다 -- 한 epoch 안에서는 어느 배치에서 보든 같은
             inter 간선이 살고, epoch 이 넘어가면 부분집합이 새로 뽑힌다.

             주의: 후보 풀을 만든 뒤에 fanout 추첨이 오므로, 후보가 이미 fanout 보다 많은
             중심에서는 간선 수가 늘지 않고 이웃 '구성'만 inter 쪽으로 섞인다. 간선이 실제로
             느는 것은 후보가 fanout 에 못 미치는 저차수 중심뿐이다.

주의: GPU feature 캐시(--gpu-cache, S13/S14)는 '어디서 읽느냐'만 바꾸므로 위 그래프
스위치와 직교한다. rtintra + 캐시를 같이 켜도 조립된 feature 는 비트 단위로 동일하다.

--- 이하 원본 docstring ---

시드 순서 실험 — '클러스터 단위 시드'가 정확도를 지키는지 확인한다.  [학습]

bench_cluster_batch 에서 A2(클러스터 단위 시드)가 products 14.5배로 압도했지만
1 epoch loss 가 나빴다(2.03 vs 0.52). 원인 후보는 배치 내 라벨 편향(시드가 같은
클러스터 -> 라벨 분포가 좁음)이다. Cluster-GCN 의 처방은 '배치당 여러 클러스터 섞기'.

여기서는 그 편향을 **청크 크기**로 제어한다:
    각 클러스터의 train 시드를 무작위로 섞고 chunk 개씩 자른 뒤,
    청크들을 전역으로 섞어 이어붙인다. 로더가 1024개씩 자르면
    배치당 클러스터 수 q ~= 1024/chunk (청크가 무작위라 서로 다른 클러스터).

    chunk=1024  q~1  (papers 기준. products 는 클러스터당 시드 ~197 이라 q~5 가 하한)
    chunk=128   q~8
    global      완전 무작위 (현재 방법 A)

학습 전제:
  - intra CSC + 매 epoch 새 노드 서브샘플 (factor, sample(epoch)) -> 랜덤성 유지
  - fanout 10/10/10, bs 1024, sage h256 L3 dropout 0.2, lr 1e-3 (기본값)
  - 평가는 원본 전체 그래프 (필터 없음)

모델 선택: 매 epoch val(과 --test-every 에 따라 test)을 재고, best 모델은 **val 기준**으로
고른다(기본, --select-by val). 학습이 끝나면 그 best 모델로 test 를 한 번 더 잰다(--screen 이 아니면
항상). 보고하는 test 정확도는 이 값이다.
--select-by test 는 test 로 고르는 옛 동작으로, test 를 모델 선택에 쓰므로 낙관 편향(leakage)이
있다 -- 학습 곡선 디버깅용이지 보고용이 아니다. 매 epoch test 를 재면 epoch 시간이 그만큼
늘어나므로(papers 는 test 214k > val 125k) 긴 학습은 --test-every N (또는 0) 으로 줄이고 마지막 test 에 맡긴다.

실행 (저장소 루트에서). 아래가 논문 측정에 쓴 전체 옵션이다 -- rtintra + node subsampling +
candidate filter + seed chunking + 두 shuffle + GPU feature cache + pipeline + direct-pinned +
late gather + GIL 해제 + part-range + rej-nodiscard:
  python main/run_falcon.py --dataset products --data-dir data/reordered/products-k1000 \
      --part-k 1000 --model sage --mode cluster --chunk 128 --epochs 30 \
      --graph rtintra --inter-keep-frac 0 --node-subsampling --factor 5 \
      --candidate-filter --seed-chunking --intra-cluster-shuffle --chunk-shuffle \
      --gpu-cache-gb 0.1 --pipeline --direct-pinned --late-gather --gil-release \
      --part-range --rej-nodiscard --tag my_run
  --dataset papers / friendster 는 --data-dir data/reordered/<papers|friendster>-k1000, --gpu-cache-gb 2.
  --gpu-cache-gb 는 feature 행렬의 일부만 GPU 에 올리는 용량 상한이다 (products 0.1 GB = 행렬의 10%,
  papers / friendster 2 GB = 3.5% / 6%). 속도만 잴 때는 --screen 을 붙이면 val/test 를 건너뛴다
  (epoch 0 은 캐시 워밍업이라 집계에서 빼고 1 epoch 이후 평균을 쓴다).

--profile 을 붙이면 epoch 안을 단계별로 쪼개 잰다(기본 경로는 건드리지 않는다):
  prep  = subsample | seed_order(lexsort/bounds/chunk/shuffle/concat) | loader_init
  train = wait(생산자 대기) | h2d | fwd | bwd | step | metric  + 워커 안의 sample/gather
  val   = 같은 분해 (bwd/step 없음)
배치마다 cuda sync 를 걸어 CPU/GPU 겹침이 끊기므로 총합은 평소보다 커진다 -- 비율을
보는 모드다. 결과는 runs/*.json 의 hist[*].prof 에도 그대로 들어간다.
"""
import argparse
import copy
import glob
import json
import os
import sys
import time
from time import perf_counter as _perf

import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'model'))
from models import build_model                                            # noqa: E402
from sampling.intra_edges import _dataset_paths, _load_csc, load_intra_csc  # noqa: E402
from sampling.keep_inter import compute_keep_inter_mask                    # noqa: E402
from sampling.masked_loader_nogil import MaskedNeighborLoader                   # noqa: E402
# [v5] 비트셋 node_mask 를 기본으로 쓴다(111MB -> 13.9MB). masked_loader_opt 가
# import 시점에 이 환경변수를 읽으므로 import 보다 먼저 세워야 한다. late-gather 를
# 쓸 때만 이 로더를 타므로 late-gather 가 꺼진 실행에는 영향이 없다.
os.environ.setdefault('FALCON_BITSET_MASK', '1')
_BITSET_ON = os.environ['FALCON_BITSET_MASK'] == '1'
from sampling.masked_loader_opt_nogil import MaskedNeighborLoader as OptLoader  # noqa: E402
from sampling.node_subsampler import ClusterNodeSubsampler                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')
OUTPUT = os.path.join(HERE, 'output')


def build_seed_order(tr_idx, pid, mode, chunk, rng, prof=None, chunk_order='shuffle',
                     intra_cluster_shuffle=None, chunk_shuffle=None):
    """이번 epoch 의 시드 순서(np.int64[Ntr])를 만든다. 모든 train 시드가 정확히 1회.

    prof 가 dict 면 단계별 시간을 채운다. 전부 CPU/numpy 라 sync 는 필요 없다.
    cluster 모드는 global(permutation 1회)보다 훨씬 무거울 수 있어서 -- 클러스터
    경계로 자르고 파이썬 리스트로 청크를 들고 흔든다 -- 어느 단계가 비싼지 본다.
    """
    def mark(k, t0):
        if prof is not None:
            prof[k] = prof.get(k, 0.0) + (_perf() - t0)
        return _perf()

    if mode == 'global':
        t0 = _perf()
        out = tr_idx[rng.permutation(tr_idx.size)]
        mark('so_perm', t0)
        return out
    # cluster: 클러스터별로 내부를 섞고 chunk 로 자른 뒤, 청크를 전역으로 섞는다
    t0 = _perf()
    # [S5-lite] 클러스터 안에서 시드를 어떤 순서로 늘어놓고 chunk 로 자르느냐.
    #   shuffle : 무작위 (현재). 청크의 시드 128개가 클러스터 전역에 흩어져 각자 다른
    #             3홉 이웃을 끌고 온다.
    #   id      : 노드 id 순. id 가 가까운 노드는 그래프상으로도 가까운 경향이 있어
    #             청크 안에서 이웃이 겹친다 -> n_id 감소 (probe_smart_chunk: -16.5%).
    #             전처리가 전혀 필요 없다. 대신 청크 내용이 매 epoch 고정된다
    #             (청크 사이 순서는 여전히 매 epoch 셔플).
    if intra_cluster_shuffle is None:
        intra_cluster_shuffle = chunk_order != 'id'
    second = rng.random(tr_idx.size) if intra_cluster_shuffle else tr_idx
    order = np.lexsort((second, pid[tr_idx]))                    # 클러스터별 정렬 + 내부 정렬
    t0 = mark('so_lexsort', t0)
    arr = tr_idx[order]
    key = pid[arr]
    bounds = np.flatnonzero(np.diff(key)) + 1                    # 클러스터 경계
    t0 = mark('so_bounds', t0)
    per_cluster = [[p[i:i + chunk] for i in range(0, p.size, chunk)]
                   for p in np.split(arr, bounds)]
    chunks = [c for cl in per_cluster for c in cl]
    t0 = mark('so_chunk', t0)
    if chunk_shuffle is False:
        pass  # Keep cluster-id order and each cluster's chunk order.
    elif chunk_order == 'grouped':
        # [아이디어 1] 크기 균형 그룹 + 라운드로빈.
        #   1) 클러스터를 청크 수 내림차순으로 정렬해 비슷한 크기끼리 q개씩 그룹으로 묶고
        #      (다중 분할을 탐욕으로 푸는 knapsack 계열 근사)
        #   2) 그룹 안에서 각 클러스터의 청크를 라운드로빈으로 배열한다.
        # 효과: 연속 배치들이 같은 q개 클러스터의 CSC/feature 영역을 재사용한다(캐시 지역성).
        #       배치당 클러스터 수는 무작위 셔플과 같게 q 근처로 유지된다.
        # 랜덤성: 같은 크기끼리의 순서·그룹 진행 순서·클러스터 내부 순서는 매 epoch 새로 뽑힌다.
        q = max(1, 1024 // chunk)
        sized = sorted(range(len(per_cluster)),
                       key=lambda i: (-len(per_cluster[i]), rng.random()))
        groups = [sized[i:i + q] for i in range(0, len(sized), q)]
        rng.shuffle(groups)
        out_chunks = []
        for g in groups:
            lists = [list(per_cluster[i]) for i in g]
            while any(lists):
                for L in lists:
                    if L:
                        out_chunks.append(L.pop(0))
        chunks = out_chunks
    elif chunk_order == 'none':
        # [ablation] 청크를 섞지 않는다 -- 클러스터 번호 순서 그대로 배치를 만든다.
        # 배치 하나가 한 클러스터에서만 나오므로 지역성은 최대지만, 배치 안의 레이블
        # 분포가 그 클러스터에 갇혀 편향된다(정확도 붕괴 지점을 보이는 대조군).
        pass
    else:
        rng.shuffle(chunks)                                      # 청크 단위 전역 셔플
    t0 = mark('so_shuffle', t0)
    out = np.concatenate(chunks)
    mark('so_concat', t0)
    if prof is not None:
        prof['so_nchunk'] = len(chunks)
    return out


class LateGather:
    """[S9] feature gather 를 워커에서 소비자 쪽으로 옮기는 래퍼.

    관찰: epoch 당 feature 72GB 가 [워커 gather-쓰기] -> [shm] -> [pin 복사] -> [H2D] 로
    네 번 DRAM 을 오간다. 이 운반 사슬이 실측 벽(wait+h2d 합 20~22초, pin 구성과 무관)이다.

    워커가 n_id/edge 만 보내면(7MB) shm 왕복 두 번이 통째로 사라진다. gather 는 여기서
    스레드풀로 한다 -- torch.index_select 는 GIL 을 놓고 내부 병렬이라 실제로 겹친다.
    이어서 pin 까지 같은 job 에서 처리해 H2D 는 pinned 로 남는다.

    direct=False  x_all -> 일반 텐서 -> pin_memory()   복사 2회
    direct=True   x_all -> pinned 에 직접              복사 1회  [S10]
        pinned 버퍼를 torch.empty(pin_memory=True) 로 매번 요청하므로 PyTorch 의
        CachingHostAllocator 를 그대로 탄다. 즉 '전송이 끝나지 않은 버퍼는 재사용하지
        않는다'는 보장이 유지된다 -- 직접 버퍼를 잡아 돌려쓰는 것과 다르다.
        (비동기 H2D 100회 반복 검증: 손상 0건, 값 동일)

    출력은 두 경로 모두 동일하다: 같은 n_id 로 같은 x_all 에서 같은 행을 뽑는다.
    """

    def __init__(self, loader, x_all, threads=4, depth=4, pin=True, direct=False,
                 h2d_device=None, pipeline=True):
        self.loader, self.x_all = loader, x_all
        self.threads, self.depth, self.pin = threads, depth, pin
        self.direct = bool(direct)
        self.pipeline = pipeline
        # [S12] h2d_device 를 주면 gather 스레드가 버퍼를 채운 '직후' 자기 스트림에서
        # H2D 까지 발행한다. 메인이 꺼내 갈 때까지 기다리지 않으므로 전송이 로더 대기
        # (실측 6.7초)와 겹친다. 스레드는 늘지 않는다 -- 같은 gather 스레드가 한다.
        self.h2d_dev = h2d_device
        self._tl = threading.local()

    def __len__(self):
        return len(self.loader)

    def _stream(self):
        """gather 스레드마다 자기 복사 스트림. 공유하면 전송이 서로 줄을 선다."""
        st = getattr(self._tl, 'st', None)
        if st is None:
            st = torch.cuda.Stream()
            self._tl.st = st
        return st

    def _job(self, n_id):
        if self.direct and self.pin:
            out = torch.empty((n_id.numel(), self.x_all.shape[1]),
                              dtype=self.x_all.dtype, pin_memory=True)
            torch.index_select(self.x_all, 0, n_id, out=out)
        else:
            x = self.x_all.index_select(0, n_id)
            out = x.pin_memory() if self.pin else x
        if self.h2d_dev is None:
            return out
        st = self._stream()
        with torch.cuda.stream(st):
            g = out.to(self.h2d_dev, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record(st)
        return (g, ev, out)          # out 을 물고 있어야 복사 전에 pinned 가 안 풀린다

    def _attach(self, bb, r, cur):
        if self.h2d_dev is None:
            bb.x = r
            return bb
        g, ev, _keep = r
        cur.wait_event(ev)           # GPU 쪽 순서만 잡는다. CPU 는 안 막힌다
        g.record_stream(cur)         # 복사 스트림에서 잡은 블록을 메인이 다 쓸 때까지 보호
        bb.x = g
        return bb

    def __iter__(self):
        if not self.pipeline:
            for b in self.loader:
                torch.cuda.synchronize()
                cur = torch.cuda.current_stream()
                b = self._attach(b, self._job(b.n_id), cur)
                torch.cuda.synchronize()
                yield b
            return

        pool = ThreadPoolExecutor(max_workers=self.threads)
        cur = torch.cuda.current_stream() if self.h2d_dev is not None else None
        pend = []
        try:
            for b in self.loader:
                pend.append((b, pool.submit(self._job, b.n_id)))
                if len(pend) >= self.depth:
                    bb, fut = pend.pop(0)
                    yield self._attach(bb, fut.result(), cur)
            while pend:
                bb, fut = pend.pop(0)
                yield self._attach(bb, fut.result(), cur)
        finally:
            pool.shutdown(wait=True)


class CountFeed:
    """[S13] 캐시 워밍업 -- epoch 0 을 그대로 돌리면서 n_id 접근 빈도만 센다."""

    def __init__(self, feed, counts):
        self.feed, self.counts = feed, counts

    def __len__(self):
        return len(self.feed)

    def __iter__(self):
        for b in self.feed:
            self.counts[b.n_id.numpy()] += 1     # 배치 안 n_id 는 유일 (merged dedup)
            yield b


class GpuCacheFeed:
    """[S13] GPU feature 캐시 -- 빈도 상위 K 노드를 GPU 에 상주시킨다.

    epoch 하나가 만지는 고유 노드는 전체의 ~4% (factor 20 서브샘플 + 이웃 샘플링의
    고차수 편중). 빈도 상위 24M 이 다음 epoch 접근의 ~82% 를 덮는다(실측). 히트분은
    gather 도 H2D 도 아예 하지 않고, 미스만 기존 경로(pinned 직행 + 즉시 발행)로
    가져와 GPU 에서 조립한다. CPU 스레드 수는 그대로다(조회는 numpy 1회, 조립은
    GPU 커널 2개). 값은 비트 단위로 동일함을 검증했다 (probe_gpu_cache).

    랭킹은 epoch 0 의 접근 빈도(CountFeed)다 -- 구조 정보만 쓰므로 레이블 누수 없음.
    """

    def __init__(self, loader, x_all, cache, map_cpu, device, stats,
                 threads=1, depth=4, direct=True, pipeline=True):
        self.loader, self.x_all = loader, x_all
        self.cache, self.map_cpu, self.dev = cache, map_cpu, device
        self.stats, self.threads, self.depth = stats, threads, depth
        self.direct, self.pipeline = direct, pipeline
        self._tl = threading.local()

    def __len__(self):
        return len(self.loader)

    def _st(self):
        st = getattr(self._tl, 'st', None)
        if st is None:
            st = torch.cuda.Stream(); self._tl.st = st
        return st

    def _job(self, n_id):
        nid = n_id.numpy()
        slot = self.map_cpu[nid]                        # int32, -1 = miss
        hit = slot >= 0
        self.stats['hits'] += int(hit.sum()); self.stats['tot'] += len(nid)
        hpos = torch.from_numpy(np.nonzero(hit)[0]).to(torch.int64).pin_memory()
        mpos = torch.from_numpy(np.nonzero(~hit)[0]).to(torch.int64).pin_memory()
        hslot = torch.from_numpy(slot[hit].astype(np.int64)).pin_memory()
        miss = torch.empty((int((~hit).sum()), self.x_all.shape[1]),
                           dtype=self.x_all.dtype, pin_memory=True)
        miss_ids = n_id[torch.from_numpy(~hit)]
        if self.direct:
            torch.index_select(self.x_all, 0, miss_ids, out=miss)
        else:
            intermediate = self.x_all.index_select(0, miss_ids)
            miss.copy_(intermediate)
        st = self._st()
        with torch.cuda.stream(st):                     # 채운 직후 발행 (S12 와 동일)
            g = (miss.to(self.dev, non_blocking=True),
                 hpos.to(self.dev, non_blocking=True),
                 mpos.to(self.dev, non_blocking=True),
                 hslot.to(self.dev, non_blocking=True))
        ev = torch.cuda.Event(); ev.record(st)
        return g, ev, (miss, hpos, mpos, hslot)         # pinned 원본 수명 유지

    def _attach(self, bb, r, cur):
        (g_miss, g_hpos, g_mpos, g_hslot), ev, _keep = r
        cur.wait_event(ev)
        for t in (g_miss, g_hpos, g_mpos, g_hslot):
            t.record_stream(cur)
        n = g_hpos.numel() + g_mpos.numel()
        x = torch.empty((n, self.x_all.shape[1]), dtype=self.x_all.dtype, device=self.dev)
        x.index_copy_(0, g_hpos, self.cache.index_select(0, g_hslot))
        x.index_copy_(0, g_mpos, g_miss)
        bb.x = x
        return bb

    def __iter__(self):
        if not self.pipeline:
            for b in self.loader:
                torch.cuda.synchronize()
                cur = torch.cuda.current_stream()
                b = self._attach(b, self._job(b.n_id), cur)
                torch.cuda.synchronize()
                yield b
            return

        pool = ThreadPoolExecutor(max_workers=self.threads)
        cur = torch.cuda.current_stream()
        pend = []
        try:
            for b in self.loader:
                pend.append((b, pool.submit(self._job, b.n_id)))
                if len(pend) >= self.depth:
                    bb, fut = pend.pop(0)
                    yield self._attach(bb, fut.result(), cur)
            while pend:
                bb, fut = pend.pop(0)
                yield self._attach(bb, fut.result(), cur)
        finally:
            pool.shutdown(wait=True)


class LruCacheFeed:
    """[S14] GPU feature 캐시 -- LRU 동적 교체 / 하이브리드(static + LRU).

    GpuCacheFeed 와 GPU 예산(K 슬롯)도, 히트/미스 조립 경로도 완전히 같다. 다른 점은
    슬롯의 주인이 고정이 아니라는 것뿐이다:

      static 영역 (앞 K_s 슬롯)  epoch 0 접근 빈도 상위 K_s 개를 고정 상주. 교체 없음.
      LRU 영역   (나머지)        미스 노드를 넣고, 가장 오래 안 쓴 슬롯을 내보낸다.

    K_s = 0 이면 순수 LRU, K_s = K 면 기존 static 캐시와 같아진다(그 경우는 GpuCacheFeed).

    왜 이득을 기대하나: epoch 하나가 만지는 고유 노드는 4.9M 인데 슬롯은 20M 이다.
    static 랭킹은 ep0 에 한 번도 안 나온 노드(count 0)까지 순위에 넣어 슬롯 대부분을
    '접근된 적 없는 노드'로 채우는 반면, LRU 는 실제로 접근된 노드만 담는다.

    정확성: 캐시는 x_all 의 사본이므로 어떤 교체 정책을 쓰든 조립된 feature 는 비트
    단위로 동일하다(정책은 '어디서 읽느냐'만 바꾼다).

    구현 메모
      · 메타데이터는 CPU(numpy), 캐시 쓰기는 GPU. 메타데이터는 잡(gather 스레드)에서
        갱신하고 GPU 쓰기는 _attach 에서 '히트 읽기 뒤'에 발행한다 -- 같은 스트림이라
        순서가 보장되므로 앞 배치가 읽는 중인 슬롯을 덮어쓰지 않는다.
      · 20M 슬롯에서 배치마다 정확한 LRU victim 을 뽑으려면 매 배치 20M argpartition 이
        필요해 너무 비싸다. victim 을 넉넉히 풀로 뽑아 여러 배치에 걸쳐 나눠 쓰고, 꺼낼
        때 '풀을 만든 뒤 다시 히트된 슬롯'은 걸러낸다(그 슬롯은 최근 사용이므로).
      · threads=1 전제 -- 잡이 순서대로 실행돼야 메타데이터 순서와 GPU 쓰기 순서가 맞는다.
    """

    def __init__(self, loader, x_all, cache, map_cpu, device, stats,
                 slot_node, last_used, n_static, threads=1, depth=4):
        self.loader, self.x_all = loader, x_all
        self.cache, self.map_cpu, self.dev = cache, map_cpu, device
        self.stats, self.threads, self.depth = stats, 1, depth   # 순서 보장 위해 1 고정
        self.slot_node, self.last_used = slot_node, last_used
        self.n_static = int(n_static)
        self.K = int(cache.shape[0])
        self._tl = threading.local()
        self._t = 0                       # recency 시계 (배치 카운터)
        self._pool = None                 # victim 후보 (오래된 순)
        self._pp = 0                      # 풀 커서
        self._stamp = -1                  # 풀을 만든 시각

    def __len__(self):
        return len(self.loader)

    def _st(self):
        st = getattr(self._tl, 'st', None)
        if st is None:
            st = torch.cuda.Stream(); self._tl.st = st
        return st

    def _refresh_pool(self, need):
        """LRU 영역에서 가장 오래 안 쓴 슬롯을 넉넉히 뽑아 둔다(비용 상각)."""
        lo, lu = self.n_static, self.last_used[self.n_static:]
        P = int(min(lu.size, max(8 * max(need, 1), 4 << 20)))
        idx = np.argpartition(lu, P - 1)[:P]
        self._pool = (idx[np.argsort(lu[idx], kind='stable')] + lo)
        self._pp, self._stamp = 0, self._t

    def _victims(self, n):
        """교체 대상 슬롯 n 개. 풀 생성 뒤 다시 히트된 슬롯은 건너뛴다."""
        out, got = [], 0
        while got < n:
            if self._pool is None or self._pp >= self._pool.size:
                self._refresh_pool(n - got)
            take = self._pool[self._pp:self._pp + 2 * (n - got)]
            self._pp += take.size
            if take.size == 0:
                continue
            ok = take[self.last_used[take] <= self._stamp]
            if ok.size:
                ok = ok[:n - got]
                out.append(ok); got += ok.size
        return out[0] if len(out) == 1 else np.concatenate(out)

    def _job(self, n_id):
        nid = n_id.numpy()
        slot = self.map_cpu[nid]                       # int32, -1 = 미적재
        hit = slot >= 0
        self.stats['hits'] += int(hit.sum()); self.stats['tot'] += len(nid)
        self._t += 1
        hs = slot[hit]
        if hs.size:
            self.last_used[hs] = self._t               # 히트 슬롯 recency 갱신
        miss_sel = ~hit
        miss_nid = nid[miss_sel]
        hpos = torch.from_numpy(np.nonzero(hit)[0]).to(torch.int64).pin_memory()
        mpos = torch.from_numpy(np.nonzero(miss_sel)[0]).to(torch.int64).pin_memory()
        hslot = torch.from_numpy(hs.astype(np.int64)).pin_memory()
        miss = torch.empty((miss_nid.size, self.x_all.shape[1]),
                           dtype=self.x_all.dtype, pin_memory=True)
        torch.index_select(self.x_all, 0, n_id[torch.from_numpy(miss_sel)], out=miss)
        # 미스 노드를 LRU 영역에 적재 -- 메타데이터만 여기서, GPU 쓰기는 _attach 에서
        vict = None
        n_lru = self.K - self.n_static
        if miss_nid.size and n_lru > 0:
            k = min(miss_nid.size, n_lru)
            v = self._victims(k)
            old = self.slot_node[v]
            om = old >= 0
            if om.any():
                self.map_cpu[old[om]] = -1             # 쫓겨난 노드는 미적재로
            ins = miss_nid[:k]
            self.slot_node[v] = ins
            self.map_cpu[ins] = v.astype(np.int32)
            self.last_used[v] = self._t
            vict = torch.from_numpy(v.astype(np.int64)).pin_memory()
        st = self._st()
        with torch.cuda.stream(st):
            g = (miss.to(self.dev, non_blocking=True),
                 hpos.to(self.dev, non_blocking=True),
                 mpos.to(self.dev, non_blocking=True),
                 hslot.to(self.dev, non_blocking=True),
                 None if vict is None else vict.to(self.dev, non_blocking=True))
        ev = torch.cuda.Event(); ev.record(st)
        return g, ev, (miss, hpos, mpos, hslot, vict)

    def _attach(self, bb, r, cur):
        (g_miss, g_hpos, g_mpos, g_hslot, g_vict), ev, _keep = r
        cur.wait_event(ev)
        for t in (g_miss, g_hpos, g_mpos, g_hslot, g_vict):
            if t is not None:
                t.record_stream(cur)
        n = g_hpos.numel() + g_mpos.numel()
        x = torch.empty((n, self.x_all.shape[1]), dtype=self.x_all.dtype, device=self.dev)
        x.index_copy_(0, g_hpos, self.cache.index_select(0, g_hslot))   # 읽기 먼저
        x.index_copy_(0, g_mpos, g_miss)
        if g_vict is not None:                                          # 그 뒤에 쓰기
            self.cache.index_copy_(0, g_vict, g_miss[:g_vict.numel()])
        bb.x = x
        return bb

    def __iter__(self):
        pool = ThreadPoolExecutor(max_workers=1)
        cur = torch.cuda.current_stream()
        pend = []
        try:
            for b in self.loader:
                pend.append((b, pool.submit(self._job, b.n_id)))
                if len(pend) >= self.depth:
                    bb, fut = pend.pop(0)
                    yield self._attach(bb, fut.result(), cur)
            while pend:
                bb, fut = pend.pop(0)
                yield self._attach(bb, fut.result(), cur)
        finally:
            pool.shutdown(wait=True)


class StreamPrefetch:
    """[S11] H2D 를 별도 CUDA 스트림으로 옮겨 다음 배치 전송을 계산 뒤에 숨긴다.

    지금은 전송이 기본 스트림이라 배치마다 H2D -> fwd -> bwd -> H2D 로 줄을 선다.
    마이크로벤치로 재보면 72GB 를 옮기는 실제 시간은 1.48초(50.5 GB/s)인데 프로파일의
    h2d 는 6.0초다 -- 나머지는 전송이 아니라 직렬화와 대역폭 경쟁이다.

    복사 스트림에서 배치 n+1 을 미리 올려두고, 메인 스트림은 이벤트만 기다린다.
    fwd+bwd+step(4.3초)과 겹치는 만큼이 그대로 이득이다.

    스레드는 늘지 않는다 -- CUDA 스트림은 GPU 쪽 큐이고, 호출하는 것은 같은 메인
    스레드다. 값도 바뀌지 않는다(같은 바이트를 같은 순서로 올린다).

    수명 주의 두 가지:
      · pinned 원본은 복사가 끝날 때까지 살아 있어야 한다 -> 배치를 pend 에 붙들어 둔다
      · 복사 스트림에서 할당한 GPU 텐서를 메인 스트림에서 쓰려면 record_stream 이
        필요하다. 안 하면 캐싱 할당자가 전송 중인 블록을 재사용할 수 있다
    """

    def __init__(self, loader, device, depth=2):
        self.loader, self.device, self.depth = loader, device, max(1, int(depth))

    def __len__(self):
        return len(self.loader)

    def _mark(self, g, cur):
        for k in ('x', 'edge_index', 'y'):
            t = getattr(g, k, None)
            if torch.is_tensor(t) and t.is_cuda:
                t.record_stream(cur)
        return g

    def __iter__(self):
        cur = torch.cuda.current_stream()
        cp = torch.cuda.Stream()
        pend = []
        for b in self.loader:
            cp.wait_stream(cur)                  # 이전 계산이 쓰던 버퍼를 침범하지 않게
            with torch.cuda.stream(cp):
                g = b.to(self.device, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(cp)
            pend.append((g, ev, b))              # b 를 잡아둬야 pinned 원본이 안 풀린다
            if len(pend) > self.depth:
                g0, ev0, _ = pend.pop(0)
                cur.wait_event(ev0)
                yield self._mark(g0, cur)
        while pend:
            g0, ev0, _ = pend.pop(0)
            cur.wait_event(ev0)
            yield self._mark(g0, cur)


class PinPrefetch:
    """[S8] pin 복사를 여러 스레드로 나누는 프리페처.

    왜 필요한가: DataLoader 의 pin_memory=True 는 shm->pinned 호스트 복사를 **스레드
    하나**가 전담한다. papers 는 배치당 x 가 61MB 라 epoch 당 72GB 를 그 한 스레드가
    옮긴다(실측 약 14초). 프로파일에서 이 시간이 'wait'(생산자 대기)로 잡혀 있어서
    지금까지 샘플링 병목으로 오인됐다. pin_memory=False 로 두면 이번엔 H2D 가
    pageable 이 되어 2.5초 -> 16초가 된다. 즉 어느 쪽이든 72GB 가 벽이다.

    여기서는 DataLoader 의 pin 을 끄고, 호스트 복사를 스레드 풀로 병렬화한다.
    tensor.pin_memory() 는 C 레벨 memcpy 라 GIL 을 놓으므로 실제로 병렬이 된다.
    H2D 는 pinned 소스에서 하므로 다시 빨라진다(약 29 GB/s).

    출력은 완전히 동일하다 -- 같은 배치를 같은 순서로 내놓고, 전송 방식만 바뀐다.
    """

    def __init__(self, loader, threads=4, depth=4):
        self.loader, self.threads, self.depth = loader, threads, depth

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        pool = ThreadPoolExecutor(max_workers=self.threads)
        pend = []
        try:
            it = iter(self.loader)
            for b in it:
                # x 가 배치 바이트의 92% 다. 나머지(edge_index/y/n_id)는 작아서 그대로 둔다.
                pend.append((b, pool.submit(lambda t: t.pin_memory(), b.x)))
                if len(pend) >= self.depth:
                    bb, fut = pend.pop(0)
                    bb.x = fut.result()
                    yield bb
            while pend:
                bb, fut = pend.pop(0)
                bb.x = fut.result()
                yield bb
        finally:
            pool.shutdown(wait=True)


_PROF_KEYS = ('iter', 'wait', 'wait_first', 'h2d', 'fwd', 'bwd', 'step', 'metric',
              'w_sample', 'w_gather', 'nb', 'nodes', 'edges', 'wall')


def new_prof():
    return dict.fromkeys(_PROF_KEYS, 0.0)


def _noop():
    pass


def run_epoch(loader, model, opt, device, train=True, prof=None, desc=None, model_name='sage'):
    """prof 가 dict 면(new_prof()) 배치 안을 단계별로 쪼개 재서 채운다.

    desc 가 문자열이면 배치 진행바를 띄운다(tqdm, stderr). 바는 leave=False 라 구간이
    끝나면 사라지고, 영구 로그는 기존의 epoch 요약 한 줄이 그대로 담당한다.

    쪼개려면 단계마다 torch.cuda.synchronize() 가 필요하다. CUDA 는 비동기라
    sync 없이 재면 forward 에는 커널 '런치' 시간만 찍히고 실제 계산은 전부
    맨 끝 D2H(float(loss)) 로 몰린다 -- 배분이 통째로 거짓말이 된다.
    대신 sync 가 CPU/GPU 겹침을 끊으므로 총합은 평상시보다 몇 % 늘어난다.
    **배분을 보는 값이지 절대 시간이 아니다** (절대값은 --profile 없이 잰 것).

    wait  = 로더가 다음 배치를 내놓기를 기다린 시간. 워커가 GPU 와 겹쳐 도니까
            이게 크면 '생산자(샘플링+gather) 병목', 작으면 'GPU 병목'이다.
    w_*   = 워커 프로세스 안에서 잰 작업량 합계(모든 워커 합). wait 과 직접
            비교하면 안 된다 -- num_workers 개가 병렬로 도므로 벽시계 기준
            점유는 대략 w_* / num_workers 다.
    """
    model.train() if train else model.eval()
    tot_loss = tot_correct = tot = 0
    nodes = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    sync = torch.cuda.synchronize if prof is not None else _noop
    # 진행바는 배치 루프 '뒤'에서만 만진다 -- next(it) 를 감싸면 tqdm 의 갱신 시간이
    # prof['wait'] 에 섞여 '생산자 병목' 판정이 흐려진다. 여기 두면 wall 의 other 로 간다.
    pbar = tqdm(total=len(loader), desc=desc, unit='b', leave=False,
                mininterval=0.5, dynamic_ncols=True) if desc else None
    t_wall = _perf()
    with ctx:
        t0 = _perf()
        it = iter(loader)                    # 여기서 워커 fork (non-persistent 로더)
        if prof is not None:
            prof['iter'] = _perf() - t0
        while True:
            t0 = _perf()
            try:
                b = next(it)
            except StopIteration:
                break
            if prof is not None:
                dt = _perf() - t0
                prof['wait'] += dt
                if prof['nb'] == 0:
                    prof['wait_first'] = dt  # 첫 배치엔 워커 콜드 스타트가 섞인다
                prof['w_sample'] += float(getattr(b, 't_sample', 0.0))
                prof['w_gather'] += float(getattr(b, 't_gather', 0.0))
                prof['edges'] += int(b.edge_index.size(1))
                prof['nb'] += 1

            t0 = _perf()
            b = b.to(device, non_blocking=True)
            sync()
            t1 = _perf()

            bs = b.batch_size
            if train:
                opt.zero_grad()
            # SAGE와 GCN 모두 hop 경계가 있으면 필요한 목적지 노드만 계산한다.
            out = model(b.x.float(), b.edge_index,
                        getattr(b, 'cum_nodes', None), getattr(b, 'cum_edges', None))
            out = out[:bs]
            y = b.y[:bs]
            loss = F.cross_entropy(out, y, ignore_index=-1)
            sync()
            t2 = _perf()

            if train:
                loss.backward()
                sync()
            t3 = _perf()
            if train:
                opt.step()
                sync()
            t4 = _perf()

            tot_loss += float(loss.detach()) * bs
            tot_correct += int((out.argmax(-1) == y).sum())
            tot += bs
            nodes += b.n_id.numel()
            if prof is not None:
                prof['h2d'] += t1 - t0
                prof['fwd'] += t2 - t1
                prof['bwd'] += t3 - t2
                prof['step'] += t4 - t3
                prof['metric'] += _perf() - t4
                prof['nodes'] += b.n_id.numel()
            if pbar is not None:
                pbar.update(1)
                # 누적 loss/acc 는 20배치마다만 문자열로 만든다. refresh=False 라 실제
                # 그리기는 tqdm 의 mininterval 이 정한다.
                if pbar.n % 20 == 0:
                    pbar.set_postfix_str(f'loss {tot_loss / max(tot, 1):.3f} '
                                         f'acc {tot_correct / max(tot, 1):.3f}', refresh=False)
    if prof is not None:
        prof['wall'] = _perf() - t_wall      # close() 전에 -- 바 정리는 측정 대상이 아니다
    if pbar is not None:
        pbar.close()
    return tot_loss / max(tot, 1), tot_correct / max(tot, 1), nodes


def fmt_prof(tag, p, workers, train=True):
    """train/val 한 구간을 두 줄로 요약한다."""
    nb = max(int(p['nb']), 1)
    acct = p['iter'] + p['wait'] + p['h2d'] + p['fwd'] + p['bwd'] + p['step'] + p['metric']
    parts = [('wait', p['wait']), ('h2d', p['h2d']), ('fwd', p['fwd'])]
    if train:
        parts += [('bwd', p['bwd']), ('step', p['step'])]
    parts += [('metric', p['metric']), ('fork', p['iter']), ('other', p['wall'] - acct)]
    body = ' '.join(f'{k} {v:6.1f}({100 * v / max(p["wall"], 1e-9):4.1f}%)' for k, v in parts)
    busy = (p['w_sample'] + p['w_gather']) / max(workers, 1)
    return (f'    {tag:<5} {p["wall"]:7.1f}s | {body}\n'
            f'    {nb:>5}b     | /batch ms: wait {1e3 * p["wait"] / nb:5.1f} '
            f'h2d {1e3 * p["h2d"] / nb:5.1f} '
            f'gpu {1e3 * (p["fwd"] + p["bwd"] + p["step"]) / nb:5.1f} '
            f'| worker sample {p["w_sample"]:6.1f}s gather {p["w_gather"]:6.1f}s '
            f'= {busy:.1f}s/{workers}w busy vs {p["wait"]:.1f}s wait '
            f'| first-batch {p["wait_first"]:.2f}s '
            f'| {p["nodes"] / nb:,.0f} n_id {p["edges"] / nb:,.0f} e /batch')


def _warm_mean(v):
    """epoch 0 은 OS 페이지캐시가 비어 있어 정상 상태가 아니다(papers 에서 sample 이 2.4배).
    그래서 평균은 epoch 0 을 뺀 '웜 평균'으로 낸다. epoch 이 하나뿐이면 그대로 쓴다."""
    if not v:
        return 0
    return round(float(np.mean(v[1:] if len(v) > 1 else v)), 3)


def _all_mean(v):
    """콜드 포함 전체 평균 (참고용)."""
    return round(float(np.mean(v)), 3) if v else 0


def save_outputs(out_dir, config, hist, best, total_time_sec):
    """meta.json / metrics.json / metrics.csv / png 3장을 (재)기록한다.

    meta.json 과 metrics.json 은 같은 내용이다 -- 결과를 읽는 도구가 둘 중 어느 이름을 찾든
    되게 두 번 쓴다. 이전 버전의 'eval_sec' 한 칸에 대응하는 것이 여기서는 val+test 라서 그 둘의 합을 넣고,
    쪼갠 값은 val_sec_list / test_sec_list 로 따로 남긴다.

    매 epoch 끝에 통째로 다시 쓴다 -> 중간에 끊겨도 완료분은 남는다.
    """
    import csv as _csv
    ep = [r['epoch'] for r in hist]
    col = lambda k: [r[k] for r in hist]                                  # noqa: E731
    eval_sec = [r['val_sec'] + r['test_sec'] for r in hist]
    meta = {
        'dataset': config['dataset'], 'model': config['model'],
        'batchsize': config['batchsize'], 'fanout': config['fanout'],
        'maxEpoch': config['maxEpoch'], 'epochs_done': len(ep),
        # 서브샘플 배율. graph=original(베이스라인)은 서브샘플이 없어 None 이다.
        # config 안에도 있지만, 런끼리 비교할 때 제일 자주 보는 값이라 최상위에 올린다.
        'factor': config['factor'], 'sample_factor': config['factor'],
        'mode': config['mode'], 'chunk': config['chunk'], 'graph': config['graph'],
        'part_k': config['part_k'], 'seed': config['seed'],
        'run_tag': config['run_tag'],
        'best_test_acc_pct': round(best['test_acc'] * 100, 4) if ep else None,
        'best_test_epoch': best['epoch'],
        'best_val_acc_pct': round(best['val_acc'] * 100, 4) if ep else None,
        'final_test_acc_list_pct': [round(a * 100, 4) for a in best['final_test']] if best.get('final_test') else None,
        'final_test_mean_pct': round(sum(best['final_test']) / len(best['final_test']) * 100, 4) if best.get('final_test') else None,
        'final_test_std_pct': (round((sum((a - sum(best['final_test']) / len(best['final_test'])) ** 2 for a in best['final_test'])
                                      / max(1, len(best['final_test']) - 1)) ** 0.5 * 100, 4) if best.get('final_test') else None),
        'total_time_sec': round(total_time_sec, 3),
        'total_time_min': round(total_time_sec / 60, 3),
        'avg_epoch_time_sec': _warm_mean(col('train_sec')),
        'avg_eval_time_sec': _warm_mean(eval_sec),
        'avg_sample_time_sec': _warm_mean(col('prep_sec')),
        'warm_from_epoch': 1 if len(ep) > 1 else 0,
        'avg_epoch_time_sec_all': _all_mean(col('train_sec')),
        'avg_eval_time_sec_all': _all_mean(eval_sec),
        'avg_sample_time_sec_all': _all_mean(col('prep_sec')),
        'epoch0_train_sec': round(hist[0]['train_sec'], 3) if ep else 0,
        'epoch0_eval_sec': round(eval_sec[0], 3) if ep else 0,
        'epoch0_prep_sec': round(hist[0]['prep_sec'], 3) if ep else 0,
        'epoch_list': ep,
        'train_loss_list': [round(v, 6) for v in col('loss')],
        'train_acc_list': [round(v * 100, 4) for v in col('train_acc')],
        'val_acc_list': [round(v * 100, 4) for v in col('val_acc')],
        'test_acc_list': [round(v * 100, 4) for v in col('test_acc')],
        'val_loss_list': [round(v, 6) for v in col('val_loss')],
        'test_loss_list': [round(v, 6) for v in col('test_loss')],
        'sample_time_list': [round(v, 3) for v in col('prep_sec')],
        'val_sec_list': [round(v, 3) for v in col('val_sec')],
        'test_sec_list': [round(v, 3) for v in col('test_sec')],
        'node_gather_list': col('node_gathers'),
        'config': config,
    }
    prep_keys = ('prep_sub_sec', 'prep_order_sec', 'prep_ki_sec',
                 'prep_ldr_sec', 'prep_wait_sec')
    for key in prep_keys:
        meta[key + '_list'] = col(key)
    for fname in ('meta.json', 'metrics.json'):
        with open(os.path.join(out_dir, fname), 'w') as f:
            json.dump(meta, f, indent=2)

    with open(os.path.join(out_dir, 'metrics.csv'), 'w', newline='') as f:
        w = _csv.writer(f)
        w.writerow(['epoch', 'train_loss', 'train_acc_pct', 'val_acc_pct', 'test_acc_pct',
                    'val_loss', 'test_loss', 'epoch_time_sec', 'eval_time_sec',
                    'val_time_sec', 'test_time_sec', 'prep_time_sec', 'node_gathers',
                    *prep_keys])
        for i, r in enumerate(hist):
            w.writerow([r['epoch'], round(r['loss'], 6), round(r['train_acc'] * 100, 4),
                        round(r['val_acc'] * 100, 4), round(r['test_acc'] * 100, 4),
                        round(r['val_loss'], 6), round(r['test_loss'], 6),
                        round(r['train_sec'], 3), round(eval_sec[i], 3),
                        round(r['val_sec'], 3), round(r['test_sec'], 3),
                        round(r['prep_sec'], 3), r['node_gathers'],
                        *(r[key] for key in prep_keys)])

    if ep:
        _draw_figs(out_dir, config, hist, ep)
    return meta


def _draw_figs(out_dir, config, hist, ep):
    """acc_curve / loss_curve / time_breakdown 3장. matplotlib 없으면 조용히 건너뛴다."""
    try:
        import matplotlib
        matplotlib.use('Agg')                       # 화면 없는 서버 -> 파일로만
        import matplotlib.pyplot as plt
    except ImportError:
        return
    from matplotlib.ticker import MaxNLocator
    col = lambda k: [r[k] for r in hist]                                  # noqa: E731
    # epoch 은 정수다 -- 기본 로케이터를 두면 0.5 눈금이 생긴다
    ints = lambda ax: ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # noqa: E731
    sub = (f"{config['model']} · {config['dataset']} · {config['graph']}"
           + (f" f{config['factor']:g}" if config['factor'] is not None else '')
           + f" · mode={config['mode']}"
           + (f" c{config['chunk']}" if config['mode'] == 'cluster' else '')
           + f" · bs {config['batchsize']} · fanout {config['fanout']} · s{config['seed']}")

    # 1) 정확도 곡선 -- 기준선은 best 를 고른 기준(기본 val)의 최고 epoch 에 긋는다
    te = [v * 100 for v in col('test_acc')]
    va = [v * 100 for v in col('val_acc')]
    pick = va if config.get('model_selection') == 'val_acc' else te
    bi = max(range(len(pick)), key=lambda i: (pick[i] == pick[i], pick[i]))   # NaN 은 뒤로
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ep, te, marker='o', linewidth=2, markersize=4, color='#ff2b83',
            markerfacecolor='none', label='test')
    ax.plot(ep, va, marker='s', linewidth=1.2, markersize=3, color='#2a78d6',
            alpha=.7, label='val')
    ax.plot(ep, [v * 100 for v in col('train_acc')], linewidth=1.0, color='#888',
            alpha=.6, label='train')
    ax.axhline(te[bi], color='#ff2b83', linestyle='--', linewidth=1.2,
               label=f'test {te[bi]:.2f}% @ep{ep[bi]} ({"val" if pick is va else "test"}-best)')
    ax.set_title(sub); ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
    ints(ax); ax.grid(True, linestyle='--', alpha=.5); ax.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'test_acc_curve.png'), dpi=150)
    plt.close(fig)

    # 2) loss 곡선 -- train/val/test 를 겹쳐 과적합 시점을 본다
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ep, col('loss'), marker='o', markersize=3, linewidth=1.6,
            color='#444', markerfacecolor='none', label='train')
    ax.plot(ep, col('val_loss'), marker='s', markersize=3, linewidth=1.4,
            color='#2a78d6', label='val')
    ax.plot(ep, col('test_loss'), marker='^', markersize=3, linewidth=1.4,
            color='#ff2b83', label='test')
    ax.set_title(f'Loss — {sub}'); ax.set_xlabel('Epoch'); ax.set_ylabel('Cross-entropy')
    ints(ax); ax.grid(True, linestyle='--', alpha=.5); ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'loss_curve.png'), dpi=150)
    plt.close(fig)

    # 3) 시간 분해 -- epoch 0 은 콜드 페이지캐시라 유독 길다(막대로 보면 바로 보인다)
    fig, ax = plt.subplots(figsize=(10, 6))
    bot = np.zeros(len(ep))
    for key, lbl, c in (('prep_sec', 'prep (subsample+order+loader)', '#9c6ade'),
                        ('train_sec', 'train', '#ff2b83'),
                        ('val_sec', 'val', '#2a78d6'),
                        ('test_sec', 'test', '#22a06b')):
        v = np.asarray(col(key), dtype=float)
        ax.bar(ep, v, bottom=bot, label=lbl, color=c)
        bot += v
    ax.set_title(f'Time breakdown — {sub}')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Seconds')
    ints(ax); ax.grid(True, axis='y', linestyle='--', alpha=.5); ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'time_breakdown.png'), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['products', 'papers', 'friendster'], default='products')

    ap.add_argument('--model', choices=['sage', 'gcn'], default='sage')
    ap.add_argument('--part-k', type=int, default=1000)
    ap.add_argument('--data-dir', type=os.path.abspath,
                    help='데이터 폴더: CSC/feature/label/split과 part_id.pth를 함께 로드')
    ap.add_argument('--node-subsampling', action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument('--candidate-filter', action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument('--intra-cluster-shuffle', action=argparse.BooleanOptionalAction,
                    default=True, help='클러스터 내부 seed 셔플; OFF는 node ID 순')
    ap.add_argument('--chunk-shuffle', action=argparse.BooleanOptionalAction,
                    default=True, help='클러스터에서 자른 chunk들의 전역 순서 셔플')
    ap.add_argument('--seed-chunking', action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument('--factor', type=float, default=5.0)
    ap.add_argument('--mode', choices=['global', 'cluster'], required=True)
    ap.add_argument('--graph', choices=['intra', 'rtintra', 'original'], default='intra',
                    help="intra = 오프라인 intra CSC + 노드 마스크. "
                         "rtintra = [S11] 원본 CSC 를 걸으면서 노드 마스크와 intra 여부를 "
                         "샘플링 때 같이 판정한다 (오프라인 intra CSC 불필요, 분포 동일). "
                         "original = baseline (원본 그래프, inter edge 살아있고 노드 마스크 없음)")
    ap.add_argument('--keep-inter-frac', type=float, default=0.0, metavar='P',
                    help='[S11-ki] keep-inter-too. >0 이면 train 노드 중 degree 하위 P '
                         '(0.1 = 하위 10%%) 에 한해 inter-cluster 간선을 살린다. '
                         '--graph rtintra 전용 -- 오프라인 intra CSC 는 inter 를 이미 지운 '
                         '파일이라 이 예외를 표현할 수 없다.')
    ap.add_argument('--keep-inter-degree', choices=['static', 'subsampled'],
                    default='subsampled',
                    help='[Q1] 하위 P 를 고를 때 쓸 degree. static = 원본 degree, 시작 시 '
                         '1회 고정. subsampled = 이번 epoch 서브샘플·intra 통과 이웃 수를 '
                         '매 epoch 재계산 (기본). 서브샘플이 매 epoch 이웃을 바꾸므로 '
                         "'이번 epoch 에 굶는 노드'는 원본 degree 로는 잡히지 않는다.")
    ap.add_argument('--inter-keep-frac', type=float, default=0.0, metavar='K',
                    help='[R] inter 이웃 무작위 보존. 중심의 이웃 중 intra 를 뺀 inter 쪽에서 '
                         '비율 K (0.2 = 20%%) 를 무작위로 살려 후보에 합류시킨다. 중심당 정확히 '
                         'K 가 아니라 기댓값이 K 인 독립 동전이다. --keep-inter-frac (저차수 '
                         '노드 전부/전무) 과 완전히 독립이며 같이 켜도 된다(OR). rtintra 전용.')
    ap.add_argument('--chunk', type=int, default=128, help='cluster 모드의 청크 크기')
    ap.add_argument('--fanouts', default='10,10,10')
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--workers', type=int, default=None,
                    help='워커 수. 생략하면 총 스레드가 공정성 규약값 9 가 되도록 '
                         '기본 8로 고정한다. '
                         '명시하면 그 값을 그대로 쓴다.')
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=None,
                    help='난수 시드. 생략하면 실행할 때마다 새로 뽑는다 -- 반복 실행이 서로 '
                         '다른 표본이 되어 평균/표준편차를 낼 수 있다. 뽑은 값은 출력과 '
                         'meta.json/runs json 에 남으므로 --seed <그 값> 으로 되돌릴 수 있다.')
    ap.add_argument('--hybrid-last', type=int, default=0,
                    help='마지막 N epoch 은 global 순서로 전환 (편향 어닐링)')
    ap.add_argument('--tag', default='',
                    help='결과를 output/<tag>/<model>/ 및 runs/<tag>/<model>/ 아래에 저장')
    ap.add_argument('--mem-opt', action='store_true',
                    help='호환 옵션: pipeline과 direct-pinned 켬. 명시적 --no-* 우선.')
    ap.add_argument('--pipeline', action=argparse.BooleanOptionalAction, default=None,
                    help='feature gather/H2D 및 다음 epoch 준비 중첩')
    ap.add_argument('--direct-pinned', action=argparse.BooleanOptionalAction, default=None,
                    help='중간 CPU 버퍼 없이 pinned 버퍼에 직접 gather')
    ap.add_argument('--late-gather', action=argparse.BooleanOptionalAction, default=True,
                    help='feature 를 main 쪽에서 n_id 로 직접 gather (기본). 끄면 기존 방식: 워커가 x[n_id] 를 '
                         '배치에 실어 보낸다. GPU 캐시는 late-gather 가 필요하므로 같이 꺼야 한다.')
    ap.add_argument('--pin-memory', action=argparse.BooleanOptionalAction, default=True,
                    help='late-gather 를 끈 경로에서 DataLoader pin_memory 사용 여부 (baseline_noopt 는 끔)')
    ap.add_argument('--gil-release', action=argparse.BooleanOptionalAction, default=True,
                    help='C++ 샘플러가 실행 중 GIL 을 놓는다 (기본). 끄면 원본처럼 GIL 을 쥔 채 돈다.')
    ap.add_argument('--part-range', action=argparse.BooleanOptionalAction, default=False,
                    help='[방법1] rtintra 의 part_id[c] 배열 조회를 파티션 id 구간 비교로 대체. '
                         'part_id 가 node id 에 단조인 reordered 데이터 전용. 샘플링 출력 비트 동일.')
    ap.add_argument('--rej-nodiscard', action=argparse.BooleanOptionalAction, default=False,
                    help='[방법2] 기각 샘플링 실패 시 결과를 버리지 않고 안 본 위치만 훑는다(+예산 확장). '
                         '분포 동일, 난수열은 달라진다.')
    ap.add_argument('--gpu-cache', type=int, default=0, metavar='K',
                    help='[S13] >0 이면 빈도 상위 K 개 노드의 feature 를 GPU 에 상주시킨다. '
                         'epoch 0 은 캐시 없이 돌며 빈도를 세고, epoch 1 부터 사용. '
                         'late-gather 필요. 스레드 증가 없음. 출력 동일.')
    ap.add_argument('--gpu-cache-gb', type=float, default=None, metavar='GB',
                    help='GPU feature 캐시 용량 상한(GB). 슬롯 수 = min(--gpu-cache, GB*1e9/(featlen*4), VRAM 여유). '
                         '--gpu-cache 없이 이것만 주면 용량만으로 정한다 (노드 수 상한 없음).')










    ap.add_argument('--test-every', type=int, default=1, metavar='N',
                    help='test 평가를 N epoch 마다(그리고 마지막 epoch 에) 한다. val 은 매 epoch. 기본 1 = 매 epoch. '
                         '0 = 학습 중에는 test 를 아예 안 잰다 (마지막 test 만 잰다; best 는 val 로 고른다). '
                         '건너뛴 epoch 의 test_acc 는 NaN 으로 기록된다.')
    ap.add_argument('--select-by', choices=['val', 'test'], default='val',
                    help='best 모델을 고르는 기준. val(기본) = val 정확도가 가장 높은 epoch; 보고하는 test 는 그 epoch 의 값. '
                         'test = test 로 고른다 (옛 동작; test 를 모델 선택에 쓰므로 낙관 편향이 있어 보고용으로 쓰면 안 된다).')
    ap.add_argument('--save-every-epoch', action='store_true',
                    help='매 epoch 끝에 model state 를 <out_dir>/ckpt_epNN.pt 로 저장한다 (런당 0.5~0.8MB x epochs). '
                         '--test-every 로 건너뛴 epoch 의 test 를 나중에 --eval-ckpts 로 채울 수 있다.')
    ap.add_argument('--eval-ckpts', default=None, metavar='DIR',
                    help='학습 대신, DIR 의 ckpt_ep*.pt 를 하나씩 불러 test(와 val) 만 평가하고 DIR/ckpt_eval.csv 에 쓴다. '
                         '다른 인자는 그 런과 같게 준다 (dataset/model/hidden/layers 만 맞으면 된다).')
    ap.add_argument('--eval-ckpt-from', type=int, default=0, metavar='E',
                    help='--eval-ckpts 에서 epoch >= E 인 체크포인트만 평가한다 (예: 26 이면 마지막 5 epoch, 0~30 기준).')
    ap.add_argument('--eval-ckpt-repeats', type=int, default=1, metavar='K',
                    help='--eval-ckpts 에서 체크포인트마다 test 를 K 번(sampling seed 를 바꿔) 평가하고 전부 기록한다.')
    ap.add_argument('--screen', action='store_true',
                    help='스크리닝: val/test 를 건너뛰고 train_sec 만 잰다 (속도 후보 선별용)')
    ap.add_argument('--profile', action='store_true',
                    help='epoch 안을 단계별로 쪼개 잰다 (배치마다 cuda sync -> 총합 몇 %% 증가)')
    ap.add_argument('--out-dir', default=None,
                    help='meta/metrics/best_model/png 를 쓸 폴더 (기본: main/output/[tag/]<model>/<시간>_seedorder_...)')
    args = ap.parse_args()
    # best 모델 선택 기준. --test-every 0 이면 학습 중 test 가 없으므로 val 로 고정한다.
    args.select_on = 'val' if (args.select_by == 'val' or args.test_every == 0) else 'test'
    if args.select_by == 'test' and args.select_on == 'val':
        print('    [select-by] --test-every 0 과 함께라 val 기준으로 고른다', flush=True)
    if args.node_subsampling is None:
        args.node_subsampling = args.graph != 'original'
    if args.candidate_filter is None:
        args.candidate_filter = args.graph != 'original'
    # [방법1] part_id 필터가 없는 구성(필터 끔, graph!=rtintra)에는 걸 대상이 없다. ablation 런처가
    # 같은 플래그로 필터 켬/끔을 모두 돌릴 수 있게 에러 대신 끄고 안내한다 -- 런 이름/config 에도
    # 실제 적용 상태가 찍히도록 이름을 만들기 전에 여기서 정리한다.
    if args.part_range and not (args.graph == 'rtintra' and args.candidate_filter):
        print('    [part-range] part_id 필터가 꺼진 구성이라 적용 대상 없음 -> 무시', flush=True)
        args.part_range = False
    if args.graph == 'original' and args.candidate_filter:
        ap.error('--candidate-filter requires --graph intra or rtintra')
    if args.seed_chunking is not None:
        args.mode = 'cluster' if args.seed_chunking else 'global'
    args.seed_chunking = args.mode == 'cluster'
    if not args.seed_chunking:
        args.intra_cluster_shuffle = False
        args.chunk_shuffle = False
    if args.seed_chunking and args.chunk <= 0:
        ap.error('--chunk must be positive')
    # Hold loader/bitset and worker count fixed across ablation axes.
    args.pipeline = args.mem_opt if args.pipeline is None else args.pipeline
    args.direct_pinned = args.mem_opt if args.direct_pinned is None else args.direct_pinned
    if args.gpu_cache_gb is not None:
        if args.gpu_cache_gb <= 0:
            ap.error('--gpu-cache-gb 는 양수여야 한다')
        if args.gpu_cache == 0:
            args.gpu_cache = 1 << 40      # 노드 수 상한 없음: 용량(GB)과 VRAM 여유로만 정한다 (build_gpu_cache 에서 N 으로 잘린다)
    if args.gpu_cache < 0:
        ap.error('--gpu-cache must be nonnegative')
    # late-gather 는 [v5] 에서 고정(1)이었다. 기존 baseline(late_gather=0, no_pin)을 같은 스크립트로
    # 재현할 수 있도록 CLI 로 풀었다. 끄면 GPU 캐시/direct-pinned/gather 중첩은 걸 대상이 없다.
    if not args.late_gather and args.gpu_cache > 0:
        ap.error('--gpu-cache 는 late-gather 가 필요하다. --no-late-gather 와 함께 쓰려면 --gpu-cache 0.')
    args.late_gather = 1 if args.late_gather else 0
    if not args.gil_release:
        os.environ['FALCON_HOLD_GIL'] = '1'       # 로더가 확장을 지연 로드할 때 읽는다 (fork 된 워커도 상속)
    args.lg_direct = args.direct_pinned
    args.lg_h2d = args.pipeline
    args.prep_ahead = args.pipeline
    if args.workers is None:
        args.workers = 7 if args.pipeline else 8
    _aux = 2 if args.pipeline else 0
    _total = args.workers + _aux

    # 기각된 최적화와 캐시 세부 튜닝은 실측상 최적인 값으로 못박는다(옵션 아님).
    args.chunk_order = 'shuffle'   # id/grouped/none 은 기각·미채택 ablation
    args.h2d_stream = 0            # 별도 CUDA 스트림 선행 H2D: 효과 0%
    args.pin_threads = 0           # pin 복사 병렬화: -3.2%
    args.no_pin = not args.pin_memory   # 기본 켬 (끄면 느려짐). baseline_noopt 재현용으로만 끈다
    args.ram_feat = False          # feature RAM 상주: -24%, 스왑 발생
    args.no_node_mask = not args.node_subsampling
    args.cache_policy = 'static'   # epoch0 빈도 상위 고정
    args.cache_headroom_gb = 4.0   # 실측 활성값 2.6GB 라 여유 충분
    args.cache_lru_extra = 0
    args.cache_rerank = 0
    args.cache_static_frac = 0.5
    args.cache_fill = 'count'      # degree 는 히트율 80.5->86.2% 인데 속도는 불변

    if args.tag and (args.tag in ('.', '..') or '/' in args.tag or '\\' in args.tag):
        ap.error('--tag는 경로 구분자 없는 단일 폴더 이름이어야 한다.')
    # train_chunk와 동일: no-node-mask는 offline intra에만 적용한다.
    no_node_mask = args.no_node_mask
    # 시드를 여기서 확정해 args 에 되박는다. 아래는 전부 args.seed 하나에서 파생되므로
    # (모델 초기화 / 노드 서브샘플 / 시드 순서 / 이웃 샘플링) 이 한 줄이 런 전체를 정한다.
    # vars(args) 가 runs json 으로, config['seed'] 가 meta.json 으로 나가니 뽑은 값은
    # 자동으로 남는다 -- '랜덤'이지만 사후 재현은 되는 상태.
    auto_seed = args.seed is None
    if auto_seed:
        args.seed = int.from_bytes(os.urandom(4), 'little')
    fan = [int(v) for v in args.fanouts.split(',')]
    dev = torch.device('cuda')
    _gtag = {'original': 'orig', 'rtintra': 'rt'}.get(args.graph, '')
    # ki 태그: 하위 비율과 degree 시점을 이름에 남긴다 (ki10b = 하위 10%, Q1=b).
    _kitag = ('' if args.keep_inter_frac <= 0 else
              f"_ki{args.keep_inter_frac * 100:g}"
              f"{'a' if args.keep_inter_degree == 'static' else 'b'}")
    # [R] 무작위 보존 태그 (r20 = inter 이웃의 20%).
    _rtag = '' if args.inter_keep_frac <= 0 else f"_r{args.inter_keep_frac * 100:g}"
    name = (f"{args.dataset}_{args.model}_{_gtag}{args.mode}" + _kitag + _rtag
            + (f"_c{args.chunk}" if args.mode == 'cluster' else '')
            + f"_s{args.seed}" + (f"_{args.tag}" if args.tag else ''))
    name += (f'_ns{int(args.node_subsampling)}_cf{int(args.candidate_filter)}'
             f'_sc{int(args.seed_chunking)}_is{int(args.intra_cluster_shuffle)}'
             f'_cs{int(args.chunk_shuffle)}_gc{args.gpu_cache if args.gpu_cache < (1 << 40) else "cap"}'
             f'_pl{int(args.pipeline)}_dc{int(args.direct_pinned)}')
    # 끈 상태(기본)에서는 기존 이름과 같게 둔다 -- 켠 런만 구분되도록 꼬리에 붙인다.
    name += ('_pr1' if args.part_range else '') + ('_nd1' if args.rej_nodiscard else '')
    name += (f'_gcgb{args.gpu_cache_gb:g}' if args.gpu_cache_gb is not None else '')   # 용량 상한을 줬을 때만
    name += (f'_te{args.test_every}' if args.test_every != 1 else '')   # 기본값이 아닐 때만
    name += ('' if args.late_gather else '_lg0') + ('' if args.pin_memory else '_pin0') + ('' if args.gil_release else '_gil1')
    runs_dir = os.path.join(RUNS, args.tag) if args.tag else RUNS
    runs_dir = os.path.join(runs_dir, args.model)
    os.makedirs(runs_dir, exist_ok=True)
    # 같은 speed tag 아래 서로 다른 factor의 실행 기록도 구분한다.
    record_name = name
    out_path = os.path.join(runs_dir, record_name + '.json')
    # graph=original 은 노드 서브샘플을 안 쓴다(g_mask=None) -> factor 가 결과에 영향이 없다.
    # v2 가 베이스라인에서 factor 를 None 으로 기록하는 것과 같은 이유로 이름에서도 뺀다.
    # rtintra 는 intra 와 같은 그래프를 다른 방법으로 만드는 것이라 factor 를 그대로 쓴다.
    factor = args.factor if args.node_subsampling else None
    # 결과 폴더: 시간 접두 -> ls 가 시간순. v2 와 같은 자리(main/output/)에 떨어뜨려
    # 같은 도구로 두 실험을 나란히 읽을 수 있게 한다. factor 는 v2 의 run_tag 와 같은
    # 표기(f5, f2.5)로 넣는다 -- factor 만 바꿔 돌린 런을 ls 로 바로 구분하려고.
    run_tag = (f'f{factor:g}_' if factor is not None else '') + name
    # tag와 모델별로 실행들을 묶는다. 명시적인 --out-dir은 기존처럼 최종 경로로 우선한다.
    output_root = os.path.join(OUTPUT, args.tag) if args.tag else OUTPUT
    output_root = os.path.join(output_root, args.model)
    out_dir = args.out_dir or os.path.join(
        output_root, f'{time.strftime("%Y%m%d_%H%M%S")}_seedorder_{run_tag}')
    os.makedirs(out_dir, exist_ok=True)

    if args.data_dir:
        csc_dir = args.data_dir
        part_id_path = os.path.join(csc_dir, 'part_id.pth')
        if os.path.exists(os.path.join(csc_dir, 'INCOMPLETE')):
            ap.error(f'전처리가 완료되지 않은 데이터입니다: {csc_dir}')
        required = ['conf.json', 'indptr.dat', 'indices.dat', 'features.dat',
                    'labels.dat', 'split_idx.pth', 'part_id.pth']
        if args.graph == 'intra' and args.candidate_filter:
            required += ['intra_csc/indptr.dat', 'intra_csc/indices.dat',
                         'intra_csc/conf.json']
        missing = [p for p in required if not os.path.isfile(os.path.join(csc_dir, p))]
        if missing:
            ap.error(f'--data-dir 필수 파일 누락 ({csc_dir}): {", ".join(missing)}. '
                     'intra 모드는 이 데이터의 새 node ID 기준 intra_csc가 필요합니다.')
    else:
        csc_dir, part_id_path = _dataset_paths(args.dataset, args.part_k)
    print(f'    데이터 -> {csc_dir}; partition -> {part_id_path}', flush=True)
    conf = json.load(open(os.path.join(csc_dir, 'conf.json')))
    N, Fdim = conf['features_shape']
    n_cls = int(conf['num_classes'])

    # 오프라인 intra CSC 는 graph=intra 일 때만 필요하다. rtintra/original 은 원본만 걷는다
    # -- 안 쓰는 8.2GB(papers) 를 매핑하지 않으면 그만큼 페이지캐시가 원본 쪽에 남는다.
    if args.graph == 'intra' and args.candidate_filter:
        iip, iidx, _ = load_intra_csc(os.path.join(os.path.dirname(part_id_path), 'intra_csc'))
        rowptr_i = torch.from_numpy(np.ascontiguousarray(iip))
        col_i = torch.from_numpy(np.asarray(iidx))
    else:
        rowptr_i = col_i = None
    oip, oidx, _ = _load_csc(csc_dir)
    rowptr_o = torch.from_numpy(np.ascontiguousarray(oip))
    col_o = torch.from_numpy(np.asarray(oidx))

    feats = np.memmap(os.path.join(csc_dir, 'features.dat'), mode='r', shape=(N, Fdim),
                      dtype=conf['features_dtype'])
    labels = np.memmap(os.path.join(csc_dir, 'labels.dat'), mode='r', shape=(N,),
                       dtype=conf['labels_dtype'])
    if args.ram_feat:
        # np.asarray 는 memmap 을 그대로 돌려준다(복사 없음). np.array 로 강제 복사해
        # 부모 프로세스에 상주시키면, fork 된 워커가 이미 매핑된 페이지를 COW 로 물려받는다.
        _t = time.time()
        _f = np.array(feats)
        print(f'    [S1] feature RAM 상주 {_f.nbytes / 1e9:.1f} GB  '
              f'({time.time() - _t:.1f}초)', flush=True)
        x_all = torch.from_numpy(_f)
    else:
        x_all = torch.from_numpy(np.asarray(feats))
    y_all = torch.from_numpy(np.asarray(labels)).to(torch.long)

    # [S13] GPU 캐시 상태. counts 는 epoch 0 워밍업에서 채워진다.
    GC = {'cache': None, 'map': None, 'hits': 0, 'tot': 0,
          'slot_node': None, 'last_used': None, 'n_static': 0,
          'counts': np.zeros(N, dtype=np.int32) if args.gpu_cache else None}

    split = torch.load(os.path.join(csc_dir, 'split_idx.pth'))
    train_m = np.zeros(N, dtype=bool); train_m[np.asarray(split['train'])] = True
    tr_idx = np.nonzero(train_m)[0]
    pid_t = torch.load(part_id_path)
    pid = pid_t.numpy() if isinstance(pid_t, torch.Tensor) else np.asarray(pid_t)
    # [S11] 샘플러에 넘길 클러스터 id. C++ 쪽 규약이 int32[N] 이다 (papers 444MB).
    # graph=rtintra 에서만 쓰므로 그 외에는 만들지 않는다.
    pid32 = (torch.from_numpy(np.ascontiguousarray(pid, dtype=np.int32))
             if args.graph == 'rtintra' and args.candidate_filter else None)
    # [방법1] 파티션 경계 int64[K+1]. 적용 대상이 없는 구성은 인자 정리 단계에서 이미 꺼졌다.
    part_bounds = None
    if args.part_range:
        if not bool((np.diff(pid) >= 0).all()):
            ap.error('--part-range 는 part_id 가 node id 에 대해 단조 증가(파티션 = 연속 id 구간)인 '
                     '데이터에서만 쓸 수 있다. reordered 데이터(--data-dir)를 쓰거나 옵션을 끈다.')
        part_bounds = torch.from_numpy(
            np.searchsorted(pid, np.arange(int(pid.max()) + 2)).astype(np.int64))
        print(f'    [part-range] 파티션 {part_bounds.numel() - 1}개를 id 구간으로 판정', flush=True)

    # [S11-ki] keep-inter-too. rtintra 전용인 것은 선택이 아니라 구조다: 오프라인 intra CSC
    # 는 inter 를 이미 물리적으로 지운 파일이라 '이 중심에서만 살린다'를 표현할 수 없고,
    # C++ 도 part_id 없이 keep_inter 를 받으면 조용히 무시한다. 그래서 조합을 여기서 막는다.
    if args.keep_inter_frac > 0 and args.graph != 'rtintra':
        ap.error('--keep-inter-frac 은 --graph rtintra 에서만 쓸 수 있다 '
                 '(part_id 필터가 있어야 예외를 걸 대상이 생긴다).')
    # [R] 같은 이유로 rtintra 전용이다. part_id 가 없으면 C++ 의 inter 판정 자체가 없어
    # 동전을 던질 대상이 생기지 않고, 조용히 무시되어 '켰는데 아무 일도 안 나는' 런이 된다.
    if not 0.0 <= args.inter_keep_frac <= 1.0:
        ap.error('--inter-keep-frac 은 0~1 사이여야 한다 (0.2 = inter 이웃의 20%).')
    if args.inter_keep_frac > 0 and args.graph != 'rtintra':
        ap.error('--inter-keep-frac 은 --graph rtintra 에서만 쓸 수 있다 '
                 '(part_id 가 있어야 어떤 이웃이 inter 인지 판정된다).')
    train_mt = torch.from_numpy(train_m)
    # static 은 원본 degree 라 epoch 에 무관하다 -> 여기서 1회. subsampled 는 keep_mask 가
    # 있어야 하므로 epoch 루프 안에서 만든다.
    ki_static = (compute_keep_inter_mask(rowptr_o, train_mt, frac=args.keep_inter_frac,
                                         degree='static', verbose=True)
                 if args.candidate_filter and args.keep_inter_frac > 0 and args.keep_inter_degree == 'static' else None)

    subsampler = (ClusterNodeSubsampler(part_id_path, N, args.factor, seed=args.seed,
                                       target_mask=train_m, verbose=False) if args.node_subsampling else None)

    # 평가 로더: 원본 전체 그래프, 필터 없음 (v2 build_eval_loaders 와 동일 전제)
    # late-gather 를 쓸 때만 skip_x 지원 로더를 쓴다. 안 쓰면 원본 로더 그대로.
    Loader = OptLoader if args.late_gather else MaskedNeighborLoader
    _vkw = {'skip_x': True} if args.late_gather else {}
    # pin-threads 를 쓰면 DataLoader 의 단일 pin 스레드는 꺼야 한다(이중 복사 방지)
    PIN = (not args.no_pin) and args.pin_threads == 0 and args.late_gather == 0
    def wrap(ld, train_feed=False):
        if args.late_gather:
            # [S13] 캐시가 준비된 뒤의 '학습' 로더만 캐시 경로를 탄다. 평가 로더는
            # 시작 시점(캐시 없음)에 만들어져 persistent 로 재사용되므로 항상 기존 경로.
            if train_feed and args.gpu_cache:
                # LRU 는 랭킹이 필요 없으므로 워밍업 없이 빈 캐시로 바로 시작한다.
                if GC['cache'] is None and args.cache_policy == 'lru':
                    build_gpu_cache()
                if GC['cache'] is not None:
                    if GC['n_static'] >= GC['cache'].shape[0]:
                        f = GpuCacheFeed(ld, x_all, GC['cache'], GC['map'], dev, GC,
                                         threads=args.late_gather, direct=args.direct_pinned,
                                         pipeline=args.pipeline)
                        # 재랭킹을 하려면 캐시가 도는 동안에도 접근 빈도를 계속 세야 한다.
                        # 그 계수 비용도 재랭킹 방식의 실제 비용이므로 측정에 포함시킨다.
                        return CountFeed(f, GC['counts']) if args.cache_rerank else f
                    return LruCacheFeed(ld, x_all, GC['cache'], GC['map'], dev, GC,
                                        GC['slot_node'], GC['last_used'], GC['n_static'],
                                        threads=args.late_gather)
            f = LateGather(ld, x_all, threads=args.late_gather, pin=PIN or True,
                           direct=args.lg_direct,
                           h2d_device=(dev if args.lg_h2d else None),
                           pipeline=args.pipeline)
            if train_feed and args.gpu_cache and GC['cache'] is None:
                return CountFeed(f, GC['counts'])
            return f
        elif args.pin_threads:
            f = PinPrefetch(ld, threads=args.pin_threads)
        else:
            f = ld
        # H2D 선행은 pinned 소스가 있어야 의미가 있다(pageable 이면 동기 복사라 안 겹친다)
        if args.h2d_stream and (args.late_gather or args.pin_threads or PIN):
            f = StreamPrefetch(f, dev, depth=args.h2d_stream)
        return f

    def build_gpu_cache():
        """GPU 캐시 슬롯 K 개를 잡는다(예산은 정책과 무관하게 동일). 여유 4GB 를 남긴다.

        static  : K 전부를 epoch 0 빈도 상위로 채우고 고정 (기존 [S13])
        hybrid  : 앞 K_s 만 빈도 상위로 채우고, 나머지는 LRU 가 채운다 [S14]
        lru     : 아무것도 미리 안 채운다 -- 워밍업 자체가 없다 [S14]
        """
        # 슬롯 수는 '최초 1회' 정한 값을 끝까지 쓴다. 재랭킹으로 다시 부를 때 free 를 새로
        # 재면 이미 잡아둔 캐시(10GB)만큼 작게 나와 K 가 붕괴한다(실측: 18.6M -> 0 -> 13.7M).
        if GC.get('K') is None:
            # [cache-fix] PyTorch 는 다 쓴 GPU 메모리를 드라이버에 돌려주지 않고 쥐고 있다(reserved). 그 상태로
            # mem_get_info() 를 재면 '쥐고만 있는' 메모리가 점유로 잡혀, epoch 0 에 allocator 가 얼마나 부풀었느냐에
            # 따라 슬롯 수가 런마다 달라졌다 (friendster/gcn 실측: 0.6M / 3.4M / 13.6M). 재기 전에 비운다.
            # 비우면 학습이 다시 쓸 메모리는 여유분에서만 나오므로, 여유는 헤드룸과 'epoch 0 최대 사용량 x 1.25'
            # 중 큰 쪽으로 잡는다 -- 사용량이 헤드룸(4GB)보다 작은 구성(papers 실측 2.6GB)에서는 헤드룸 그대로다.
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info()
            reserve = max(args.cache_headroom_gb * 1e9, 1.25 * peak)
            room = max(0, free - reserve)
            cap = int(args.gpu_cache_gb * 1e9 // (Fdim * 4)) if args.gpu_cache_gb is not None else N   # 용량 상한 -> 슬롯 수
            GC['K'] = int(min(N, args.gpu_cache, cap, room // (Fdim * 4)))
            req = f'{args.gpu_cache / 1e6:.0f}M' if args.gpu_cache < (1 << 40) else '상한 없음'
            say(f'    [cache-fix] free {free / 1e9:.2f}GB · epoch0 peak {peak / 1e9:.2f}GB · reserve {reserve / 1e9:.2f}GB '
                f'-> {GC["K"] / 1e6:.2f}M 슬롯 = {GC["K"] * Fdim * 4 / 1e9:.2f}GB (요청 {req}'
                + (f', 용량 상한 {args.gpu_cache_gb:g}GB = {cap / 1e6:.2f}M' if args.gpu_cache_gb is not None else '') + ')')
            # static 을 줄이지 않고 '남는 VRAM' 에만 LRU 를 얹는 구성. 예산을 반으로 쪼개는
            # hybrid 와 달리 static 커버리지는 그대로 두고 미스 보존만 추가한다.
            if args.cache_lru_extra and args.cache_policy == 'static':
                # static 은 기본 헤드룸 기준 크기를 그대로 유지하고, 그 헤드룸(놀고 있는 VRAM,
                # 실측 3.9GB)에서 LRU 슬롯을 더 가져온다. 안전 여유 0.8GB 만 남긴다.
                GC['n_static_fixed'] = GC['K']
                hard = int(max(0, free - 0.8e9) // (Fdim * 4))
                GC['K'] = int(min(GC['K'] + args.cache_lru_extra, hard))
        K = GC['K']
        pol = args.cache_policy
        if GC.get('n_static_fixed') is not None:
            n_s = GC['n_static_fixed']
        else:
            n_s = K if pol == 'static' else (int(K * args.cache_static_frac) if pol == 'hybrid' else 0)
        m = np.full(N, -1, dtype=np.int32)
        slot_node = np.full(K, -1, dtype=np.int64)
        last_used = np.full(K, -1, dtype=np.int32)
        cache = (GC['cache'] if GC['cache'] is not None and GC['cache'].shape[0] == K
                 else torch.empty((K, Fdim), dtype=x_all.dtype, device=dev))
        t0 = time.time()
        if n_s:
            if args.cache_fill == 'degree':
                # count>0 을 먼저 전부 넣고, 남는 슬롯만 intra 차수 상위로 채운다.
                pos = np.flatnonzero(GC['counts'] > 0)
                if pos.size >= n_s:
                    ids = pos[np.argpartition(GC['counts'][pos], -n_s)[-n_s:]]
                else:
                    rest = n_s - pos.size
                    _rp = rowptr_i if args.graph == 'intra' and args.candidate_filter else rowptr_o
                    deg = (_rp[1:] - _rp[:-1]).numpy()
                    deg = deg.copy()
                    deg[pos] = -1                      # 이미 넣은 것은 후보에서 제외
                    top = np.argpartition(deg, -rest)[-rest:]
                    ids = np.concatenate([pos, top])
                    say(f'    [채움] count>0 {pos.size/1e6:.2f}M + 차수상위 '
                        f'{rest/1e6:.2f}M')
            else:
                ids = np.argpartition(GC['counts'], -n_s)[-n_s:]
            ids = np.sort(ids)                          # id 순 -> fill 지역성
            m[ids] = np.arange(n_s, dtype=np.int32)
            slot_node[:n_s] = ids
            step = 2_000_000
            for i in range(0, n_s, step):
                sel = torch.from_numpy(ids[i:i + step].astype(np.int64))
                cache[i:i + sel.numel()].copy_(x_all.index_select(0, sel))
        GC['cache'], GC['map'] = cache, m
        GC['slot_node'], GC['last_used'], GC['n_static'] = slot_node, last_used, n_s
        # 구성/재랭킹 때 자동 출력. 진단은 캐시 선택과 RNG를 변경하지 않는다.
        c = GC['counts']
        ct, ca = c[tr_idx], c[c > 0]
        n_in = int((m[tr_idx] >= 0).sum())
        say(f'    [진단] 접근>0 노드 {ca.size/1e6:.2f}M · 학습 노드 '
            f'{tr_idx.size/1e6:.2f}M 중 캐시 적재 {n_in/1e6:.2f}M '
            f'({n_in/max(tr_idx.size, 1)*100:.2f}%)')
        for label, values in (('학습노드', ct), ('접근>0 전체', ca)):
            if values.size:
                say(f'    [진단] 접근 횟수 분위 — {label} ' +
                    ' '.join(f'p{q}={np.percentile(values, q):.0f}'
                             for q in (50, 75, 90, 99)) + f' 평균 {values.mean():.1f}')
            else:
                say(f'    [진단] 접근 횟수 분위 — {label}: 표본 없음')
        if ca.size:
            # 빈도 내림차순, 동률은 노드 ID 순: 원본 stable argsort와 동일.
            # N 크기 순위 배열 두 개를 만들지 않고 빈도별 개수로 경계를 찾는다.
            vals, nums = np.unique(c, return_counts=True)
            cumulative = np.cumsum(nums[::-1])
            for budget in (1_000_000, 2_000_000, 5_000_000):
                limit = min(budget, N)
                j = int(np.searchsorted(cumulative, limit))
                threshold = vals[-1-j]
                above = int(cumulative[j-1]) if j else 0
                remaining = limit - above
                # 동률 ID는 블록 단위로 훑어 임시 메모리를 제한한다.
                cutoff = -1
                for start in range(0, N, 1_000_000):
                    tied = np.flatnonzero(c[start:start+1_000_000] == threshold)
                    if remaining <= tied.size:
                        cutoff = start + int(tied[remaining-1])
                        break
                    remaining -= tied.size
                included = int(((ct > threshold) |
                                ((ct == threshold) & (tr_idx <= cutoff))).sum())
                say(f'    [진단] 빈도 상위 {budget/1e6:.0f}M 만 캐싱하면 학습 노드 '
                    f'{included/max(tr_idx.size, 1)*100:5.1f}% 포함')
        else:
            say('    [진단] 빈도별 캐시 크기 비교: 접근 기록 없음 (워밍업 전)')
        rp = rowptr_i if args.graph == 'intra' and args.candidate_filter else rowptr_o
        deg = (rp[1:] - rp[:-1]).numpy()
        filled = (m >= 0) & (c == 0)
        def mean_or_zero(values):
            return float(values.mean()) if values.size else 0.0
        say(f'    [진단] 강제 채움 {int(filled.sum())/1e6:.2f}M 개 · '
            f'평균 차수 {mean_or_zero(deg[filled]):.2f} · '
            f'전체 평균 {mean_or_zero(deg):.2f} · '
            f'접근된 노드 평균 {mean_or_zero(deg[c > 0]):.2f}')
        if filled.any():
            say(f'    [진단] 강제 채움 노드 id 중앙값 '
                f'{np.median(np.flatnonzero(filled)):,.0f} / 전체 {N:,}')
        say(f'    [{"S13" if pol == "static" else "S14"}] GPU 캐시 {K / 1e6:.1f}M 슬롯 '
            f'({K * Fdim * 4 / 1e9:.2f}GB) · 정책 {pol} · static {n_s / 1e6:.1f}M / '
            f'LRU {(K - n_s) / 1e6:.1f}M · 적재 {time.time() - t0:.1f}초')

    def eval_loader(mask):
        seeds = torch.from_numpy(np.asarray(mask)).to(torch.int64)
        return Loader(rowptr_o, col_o, seeds, fan, x_all, y_all,
                                    node_mask=None, part_id=None,
                                    batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.workers, pin_memory=PIN,
                                    persistent=True, profile=args.profile, **_vkw)
    # 둘 다 persistent -- 매 epoch 재사용하므로 워커 fork 비용을 한 번만 낸다.
    val_loader = None if args.screen else wrap(eval_loader(split['valid']))
    test_loader = None if args.screen else wrap(eval_loader(split['test']))

    torch.manual_seed(args.seed)
    model = build_model(args.model, Fdim, args.hidden, n_cls, args.layers,
                        dropout=args.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f'=== {name}: {args.epochs} epochs, fanout {fan}, lr {args.lr} ===', flush=True)
    print(f'    seed {args.seed}'
          + ('  (자동 생성 -- 재현하려면 --seed %d)' % args.seed if auto_seed
             else '  (--seed 로 지정)'), flush=True)
    print(f'    출력 -> {out_dir}', flush=True)
    print(f'    [옵션] gpu_cache={args.gpu_cache if args.gpu_cache < (1 << 40) else "cap"}'
          f'{f" (<= {args.gpu_cache_gb:g}GB)" if args.gpu_cache_gb is not None else ""}, pipeline={args.pipeline}, '
          f'direct_pinned={args.direct_pinned}', flush=True)
    print(f'    [실행] loader workers={args.workers}, gather threads='
          f'{int(args.pipeline)}, prep threads={int(args.pipeline)}; '
          'loader sampling prefetch는 유지', flush=True)

    config = {
        'data_dir': os.path.abspath(csc_dir),
        'part_id_path': os.path.abspath(part_id_path),
        'dataset': args.dataset, 'model': args.model, 'batchsize': args.batch_size,
        'gcn_execution': 'full_norm_hop_prefix' if args.model == 'gcn' else None,
        'fanout': fan, 'eval_fanout': fan, 'maxEpoch': args.epochs,
        'mode': args.mode, 'chunk': args.chunk if args.mode == 'cluster' else None,
        'chunk_order': args.chunk_order,
        'graph': args.graph, 'factor': factor, 'part_k': args.part_k,
        'hybrid_last': args.hybrid_last, 'run_tag': run_tag,
        'lr': args.lr, 'seed': args.seed, 'hidden': args.hidden,
        'num_layers': args.layers, 'dropout': args.dropout,
        'num_workers': args.workers, 'pin_memory': not args.no_pin,
        'pin_threads': args.pin_threads, 'late_gather': args.late_gather,
        'lg_direct': args.lg_direct, 'h2d_stream': args.h2d_stream,
        'lg_h2d': args.lg_h2d, 'gpu_cache': args.gpu_cache, 'gpu_cache_gb': args.gpu_cache_gb,
        'mem_opt': args.mem_opt, 'total_threads': _total,
        'pipeline': args.pipeline, 'direct_pinned': args.direct_pinned,
        'node_subsampling': args.node_subsampling,
        'candidate_filter': args.candidate_filter, 'seed_chunking': args.seed_chunking,
        'part_range': part_bounds is not None, 'rej_nodiscard': args.rej_nodiscard,
        'gil_release': args.gil_release,
        'intra_cluster_shuffle': args.intra_cluster_shuffle,
        'chunk_shuffle': args.chunk_shuffle,
        'cache_policy': args.cache_policy, 'cache_static_frac': args.cache_static_frac,
        'no_node_mask': args.no_node_mask, 'prep_ahead': args.prep_ahead,
        'cache_fill': args.cache_fill,
        'bitset_mask': bool(args.late_gather) and _BITSET_ON,
        'keep_inter_frac': args.keep_inter_frac,
        'keep_inter_degree': args.keep_inter_degree if args.keep_inter_frac > 0 else None,
        'inter_keep_frac': args.inter_keep_frac,
        'chunk_order': args.chunk_order,
        'num_nodes': N, 'featlen': Fdim, 'classes': n_cls,
        'framework': 'pyg+cpp_masked_sampler', 'version': 'seed_order',
        'eval_graph': 'original_full', 'eval_method': 'fanout_sampled',
        'model_selection': args.select_on + '_acc',   # 'val_acc' (기본) / 'test_acc' (--select-by test; leakage 있음)
        'test_every': args.test_every, 'save_every_epoch': args.save_every_epoch,
    }
    if args.eval_ckpts:                                   # 저장된 체크포인트의 test 평가만 하고 끝낸다
        import csv as _csv
        ckpts = sorted(glob.glob(os.path.join(args.eval_ckpts, 'ckpt_ep*.pt')))
        ckpts = [c for c in ckpts if int(os.path.basename(c)[7:9]) >= args.eval_ckpt_from]
        if not ckpts:
            raise SystemExit(f'--eval-ckpts: {args.eval_ckpts} 에 epoch>={args.eval_ckpt_from} 인 ckpt_ep*.pt 가 없다')
        seeds_te = torch.from_numpy(np.asarray(split['test'])).to(torch.int64)
        rows = []
        for ck in ckpts:
            st_ = torch.load(ck, map_location=dev)
            model.load_state_dict(st_['model_state']); model.to(dev)
            t0 = time.time(); vloss, vacc, _ = run_epoch(val_loader, model, None, dev, train=False, model_name=args.model); t_val = time.time() - t0
            tests = []
            for k in range(args.eval_ckpt_repeats):          # 반복마다 새 loader + 다른 seed -> 이웃 sampling 이 달라진다
                torch.manual_seed(args.seed * 31 + k + 1)
                ld = wrap(Loader(rowptr_o, col_o, seeds_te, fan, x_all, y_all, node_mask=None, part_id=None, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.workers, pin_memory=PIN, persistent=False, profile=False, **_vkw))
                t0 = time.time(); teloss, teacc, _ = run_epoch(ld, model, None, dev, train=False, model_name=args.model); t_test = time.time() - t0
                tests.append(teacc)
            rows.append([st_['epoch'], round(vacc * 100, 4), round(max(tests) * 100, 4), round(sum(tests) / len(tests) * 100, 4),
                         ' '.join(f'{t * 100:.4f}' for t in tests), round(vloss, 4), round(t_val, 2), round(t_test, 2)])
            print(f'  ckpt ep{st_["epoch"]:02d}: val {vacc:.4f} test ' + ' '.join(f'{t:.4f}' for t in tests) + f'  ({t_test:.0f}s/회)', flush=True)
        out = os.path.join(args.eval_ckpts, f'ckpt_eval_from{args.eval_ckpt_from}_x{args.eval_ckpt_repeats}.csv')
        with open(out, 'w', newline='') as f:
            w = _csv.writer(f); w.writerow(['epoch', 'val_acc_pct', 'test_acc_max_pct', 'test_acc_mean_pct', 'test_acc_list_pct', 'val_loss', 'val_time_sec', 'test_time_sec']); w.writerows(rows)
        best_row = max(rows, key=lambda r: r[2])
        print(f'  best test {best_row[2]:.4f} (ep {best_row[0]})  -> {out}', flush=True)
        return
    hist = []
    best = {'test_acc': -1.0, 'val_acc': -1.0, 'epoch': -1, 'state': None}
    t_start = time.time()
    # 바는 tty 여부와 무관하게 항상 띄운다. 리다이렉트하면 로그에 \r 갱신이 그대로
    # 남지만, epoch 요약 한 줄도 같이 남으므로 결과를 읽는 데는 지장이 없다.
    ep_bar = tqdm(total=args.epochs, desc='epochs', unit='ep', dynamic_ncols=True)

    def say(msg):
        """진행바가 떠 있으니 tqdm.write 로 -- 그냥 print 하면 바를 뭉갠다."""
        tqdm.write(msg)

    def _prep(ep):
        pp = {} if args.profile else None
        # original 경로는 아래에서 node_mask를 넘기지 않으므로 서브샘플
        # 결과를 전혀 사용하지 않는다. baseline prep 시간에 불필요한 비용이
        # 섞이지 않도록 해당 경로에서는 호출 자체를 건너뛴다.
        if not args.node_subsampling:
            sub = None
            t_sub = 0.0
        else:
            t0 = time.time()
            sub = subsampler.sample(ep)                      # 매 epoch 새 서브샘플 (랜덤성)
            t_sub = time.time() - t0
        # [S11-ki] Q1=b 는 이번 서브샘플에 의존하므로 여기서 다시 만든다. 세는 값은
        # '이 중심에서 alive() 가 bypass 없이 찾아낼 후보 수' = keep_mask 통과 + intra 이웃.
        # 원본 degree 가 높아도 이웃이 죄다 탈락했으면 하위로 내려온다.
        t0 = time.time()
        if not args.candidate_filter or args.keep_inter_frac <= 0:
            ki = None
        elif args.keep_inter_degree == 'static':
            ki = ki_static
        else:
            ki = compute_keep_inter_mask(
                rowptr_o, train_mt, frac=args.keep_inter_frac, degree='subsampled',
                col=col_o, keep_mask=(sub.keep_mask if sub is not None else np.ones(N, dtype=bool)), part_id=pid32, verbose=False)
        t_ki = time.time() - t0
        t0 = time.time()
        rng = np.random.default_rng(args.seed * 100003 + ep)
        mode_ep = ('global' if args.hybrid_last and ep >= args.epochs - args.hybrid_last
                   else args.mode)
        order_np = build_seed_order(tr_idx, pid, mode_ep, args.chunk, rng, prof=pp,
                                    chunk_order=args.chunk_order,
                                    intra_cluster_shuffle=args.intra_cluster_shuffle,
                                    chunk_shuffle=args.chunk_shuffle)
        seeds = torch.from_numpy(order_np).to(torch.int64)
        t_order = time.time() - t0
        return sub, ki, seeds, t_sub, t_ki, t_order, pp

    # context manager는 예외/중단 시에도 보조 스레드를 회수한다.
    with ExitStack() as prep_stack:
        prep_pool = (prep_stack.enter_context(ThreadPoolExecutor(max_workers=1))
                     if args.prep_ahead else None)
        prep_future = (prep_pool.submit(_prep, 0)
                       if prep_pool is not None and args.epochs > 0 else None)
        if prep_pool is not None:
            say('    [S16] prep 선발주 켜짐 — 보조 스레드 1개, 다음 epoch 준비와 학습 중첩')
        for ep in range(args.epochs):
            d_tr, d_va, d_te = f'ep{ep} train', f'ep{ep} val', f'ep{ep} test'
            pp = {} if args.profile else None                    # prep 단계 세부
            t_wait = time.time()
            if prep_pool is None:
                sub, ki, seeds, t_sub, t_ki, t_order, pp = _prep(ep)
                t_wait = 0.0
            else:
                sub, ki, seeds, t_sub, t_ki, t_order, pp = prep_future.result()
                t_wait = time.time() - t_wait
                prep_future = (prep_pool.submit(_prep, ep + 1)
                               if ep + 1 < args.epochs else None)
            t0 = time.time()
            # 세 경로 모두 같은 C++ 샘플러/로더를 탄다. 다른 것은 '어떤 CSC 를 걷고 어떤 필터를
            # 켜느냐'뿐이다:
            #   intra   오프라인 intra CSC + node_mask        (inter 는 전처리에서 이미 제거됨)
            #   rtintra 원본 CSC + node_mask + part_id        [S11] intra 판정을 샘플링 때 같이
            #   original 원본 CSC + 필터 없음                  (baseline)
            # intra 와 rtintra 는 같은 간선 집합을 만든다 -- alive() 가 두 필터를 교집합으로
            # 판정하고(masked_sampler.cpp), 서브샘플은 part_id 를 바꾸지 않기 때문이다.
            # 다만 기각 샘플링이 '행 길이' 기준으로 위치를 뽑아 RNG 스트림이 갈리므로, 뽑히는
            # 이웃까지 같지는 않다(분포만 같다). fanout 이 전부 -1 이면 무작위성이 없어져
            # 두 경로가 비트 단위로 일치해야 한다 -- 필터 등가성 검증은 그 설정으로.
            if args.graph == 'intra' and args.candidate_filter:
                g_rp, g_col, g_mask, g_pid = (rowptr_i, col_i,
                                                None if sub is None else sub.keep_mask, None)
            elif args.graph == 'rtintra' and args.candidate_filter:
                g_rp, g_col, g_mask, g_pid = rowptr_o, col_o, (None if sub is None else sub.keep_mask), pid32
            else:
                g_rp, g_col, g_mask, g_pid = rowptr_o, col_o, (None if sub is None else sub.keep_mask), None
            loader = Loader(g_rp, g_col, seeds, fan, x_all, y_all,
                            node_mask=g_mask, part_id=g_pid, keep_inter=ki,
                            part_bounds=(part_bounds if g_pid is not None else None),
                            rej_nodiscard=args.rej_nodiscard,
                            inter_keep_frac=(args.inter_keep_frac if args.candidate_filter else 0.0),
                            # [R] epoch 마다 다른 씨앗 -> 이번 epoch 에 살아남을 inter 부분집합이
                            # 새로 뽑힌다. 한 epoch 안에서는 모든 배치가 같은 집합을 본다.
                            # 상수가 seed_order 의 rng(100003)와 다른 것은 두 난수열을
                            # 겹치지 않게 하려는 것이다.
                            inter_keep_seed=args.seed * 7919 + ep,
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=PIN,
                            profile=args.profile, **_vkw)
            t_ldr = time.time() - t0
            # prep_sec는 노출된 시간; 각 단계의 실제 계산 시간은 별도 보존.
            t_prep = (t_wait if args.prep_ahead else t_sub + t_ki + t_order) + t_ldr

            tp = new_prof() if args.profile else None
            t0 = time.time()
            GC['hits'] = GC['tot'] = 0
            loss, tacc, nodes = run_epoch(wrap(loader, train_feed=True), model, opt, dev,
                                          train=True, prof=tp, desc=d_tr, model_name=args.model)
            t_train = time.time() - t0
            del loader
            # 히트율은 빌드/재랭킹 분기와 무관하게 항상 남긴다 -- 정책을 가르는 핵심 지표다.
            cache_hit = round(GC['hits'] / GC['tot'] * 100, 2) if GC['tot'] else None
            if cache_hit is not None:
                say(f'    [캐시] hit {cache_hit:.1f}%')
            if args.gpu_cache and GC['cache'] is None and ep == 0:
                build_gpu_cache()                       # static/hybrid: epoch 1 부터 캐시 경로
                if args.cache_rerank:
                    GC['counts'][:] = 0                 # 다음 구간은 '직전 epoch' 만 센다
            elif args.gpu_cache and args.cache_rerank and (ep + 1) % args.cache_rerank == 0:
                # 직전 구간의 접근 빈도로 다시 랭킹한다. 재적재 시간은 아래 t_rerank 로 따로 잰다.
                _t0 = time.time()
                build_gpu_cache()
                t_rerank = time.time() - _t0
                GC['counts'][:] = 0                     # 다음 구간은 새로 센다
                say(f'    [재랭킹] ep{ep} 기준으로 캐시 재구성 {t_rerank:.1f}초')


            if args.screen:                       # 속도만 본다 -- 평가 생략
                vp = sp = None
                vloss = vacc = teloss = teacc = 0.0
                t_val = t_test = 0.0
            else:
                vp = new_prof() if args.profile else None
                t0 = time.time()
                vloss, vacc, _ = run_epoch(val_loader, model, None, dev, train=False, prof=vp, desc=d_va, model_name=args.model)
                t_val = time.time() - t0

                do_test = (args.test_every == 1) or (args.test_every > 1 and (ep % args.test_every == 0 or ep == args.epochs - 1))
                if do_test:
                    sp = new_prof() if args.profile else None
                    t0 = time.time()
                    teloss, teacc, _ = run_epoch(test_loader, model, None, dev, train=False, prof=sp, desc=d_te, model_name=args.model)
                    t_test = time.time() - t0
                else:                                 # 이번 epoch 은 test 생략 (--test-every)
                    sp = None; teloss = teacc = float('nan'); t_test = 0.0

            # best 선택: 기본은 val (leakage 없음). --select-by test 는 test 로 고른다 (test 를 잰 epoch 에서만 갱신).
            improved = (vacc > best['val_acc']) if args.select_on == 'val' else (teacc == teacc and teacc > best['test_acc'])
            if improved:
                best.update(test_acc=teacc, val_acc=vacc, epoch=ep,
                            state=copy.deepcopy(model.state_dict()))
                torch.save({'model_state': best['state'], 'config': config, 'epoch': ep,
                            'test_acc': teacc, 'val_acc': vacc},
                           os.path.join(out_dir, 'best_model.pt'))
            if args.save_every_epoch:
                torch.save({'model_state': model.state_dict(), 'epoch': ep, 'val_acc': vacc, 'test_acc': teacc},
                           os.path.join(out_dir, f'ckpt_ep{ep:02d}.pt'))
            rec = dict(epoch=ep, loss=round(loss, 4), train_acc=round(tacc, 4),
                       val_acc=round(vacc, 4), test_acc=round(teacc, 4),
                       val_loss=round(vloss, 4), test_loss=round(teloss, 4),
                       prep_sec=round(t_prep, 2),
                       prep_sub_sec=round(t_sub, 2), prep_order_sec=round(t_order, 2),
                       prep_ki_sec=round(t_ki, 2), prep_ldr_sec=round(t_ldr, 2),
                       prep_wait_sec=round(t_wait, 2),
                       train_sec=round(t_train, 2), val_sec=round(t_val, 2),
                       test_sec=round(t_test, 2), node_gathers=nodes,
                       cache_hit=cache_hit,
                       cache_slots=GC.get('K'))        # 실제로 잡힌 GPU 캐시 슬롯 수 (논문 기재용)
            if args.profile:
                rec['prof'] = dict(
                    prep=dict(subsample=round(t_sub, 3), keep_inter=round(t_ki, 3),
                              seed_order=round(t_order, 3), loader_init=round(t_ldr, 3),
                              **{k: (round(v, 3) if isinstance(v, float) else v)
                                 for k, v in pp.items()}),
                    train={k: round(v, 3) for k, v in tp.items()},
                    val={k: round(v, 3) for k, v in vp.items()} if vp else {},
                    test={k: round(v, 3) for k, v in sp.items()} if sp else {})
            hist.append(rec)
            say(f'  ep {ep:>2}  loss {loss:.4f}  train {tacc:.4f}  val {vacc:.4f}'
                f'  test {teacc:.4f}'
                f'   (prep {t_prep:.1f}s train {t_train:.1f}s val {t_val:.1f}s test {t_test:.1f}s'
                f'  n_id/batch {nodes / max(len(seeds) // args.batch_size, 1):,.0f})')
            if args.profile:
                det = ' '.join(f'{k[3:]} {v:.2f}' for k, v in pp.items()
                               if k.startswith('so_') and k != 'so_nchunk')
                say(f'    prep  {t_prep:7.1f}s | subsample {t_sub:.2f} '
                    + (f'keep_inter {t_ki:.2f} ' if args.keep_inter_frac > 0 else '')
                    + f'seed_order {t_order:.2f} ({det}) loader_init {t_ldr:.2f}'
                    + (f'  [{pp["so_nchunk"]:,} chunks]' if 'so_nchunk' in pp else ''))
                say(fmt_prof('train', tp, args.workers, train=True))
                if vp: say(fmt_prof('val', vp, args.workers, train=False))
                if sp: say(fmt_prof('test', sp, args.workers, train=False))
            # 남은 시간은 tqdm 이 epoch 소요시간으로 직접 낸다. postfix 는 지금 성적.
            ep_bar.set_postfix_str(f'best val {max(best["val_acc"], 0):.4f} '
                                   f'test {best["test_acc"]:.4f} @ep{best["epoch"]}', refresh=False)
            ep_bar.update(1)
            # 둘 다 매 epoch 통째로 다시 쓴다 -> 중간에 끊겨도 완료분은 남는다.
            json.dump(dict(args=vars(args), hist=hist, best_test=best['test_acc'],
                           best_val=best['val_acc'], best_epoch=best['epoch'],
                           select_on=args.select_on),
                      open(out_path, 'w'), indent=1)
            save_outputs(out_dir, config, hist, best, time.time() - t_start)

    ep_bar.close()

    if best['state'] is not None and not args.screen:   # --screen 은 평가 없음
        # 학습이 끝나면 best 모델(기본 val 기준)로 test 를 한 번 더 잰다. 보고하는 test 정확도는 이 값이다.
        model.load_state_dict(best['state']); model.to(dev)
        seeds_te = torch.from_numpy(np.asarray(split['test'])).to(torch.int64)
        torch.manual_seed(args.seed * 31 + 1)
        ld = wrap(Loader(rowptr_o, col_o, seeds_te, fan, x_all, y_all, node_mask=None, part_id=None,
                         batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=PIN,
                         persistent=False, profile=False, **_vkw))
        t0 = time.time()
        _, acc_final, _ = run_epoch(ld, model, None, dev, train=False, desc='final test', model_name=args.model)
        print(f'  final test (val-best ep{best["epoch"]}): {acc_final:.4f}  ({time.time() - t0:.0f}s)', flush=True)
        accs = [acc_final]
        best['final_test'] = accs
        best['test_acc'] = acc_final
        torch.save({'model_state': best['state'], 'config': config, 'epoch': best['epoch'],
                    'test_acc': best['test_acc'], 'val_acc': best['val_acc'], 'final_test': accs},
                   os.path.join(out_dir, 'best_model.pt'))
        save_outputs(out_dir, config, hist, best, time.time() - t_start)

    # best['state'] 는 best_model.pt 로 이미 저장돼 있고, best['test_acc'] 는 그 epoch 의 test 값
    # (--screen 이 아니면 위에서 마지막에 다시 잰 값) 이다.
    json.dump(dict(args=vars(args), hist=hist, best_test=best['test_acc'],
                   best_val=best['val_acc'], best_epoch=best['epoch'],
                   test_acc=best['test_acc'], select_on=args.select_on),
              open(out_path, 'w'), indent=1)
    if best.get('final_test'):
        print(f'\n  val-best ep {best["epoch"]} (val {best["val_acc"]:.4f}) -> final test {best["test_acc"]:.4f}', flush=True)
    elif args.select_on == 'val':
        print(f'\n  val-best ep {best["epoch"]} (val {best["val_acc"]:.4f}) -> test {best["test_acc"]:.4f}'
              + ('  [그 epoch 에 test 를 안 쟀음]' if best['test_acc'] != best['test_acc'] else ''),
              flush=True)
    else:
        print(f'\n  best test {best["test_acc"]:.4f} (ep {best["epoch"]}, '
              f'그때 val {best["val_acc"]:.4f})  [test 로 고른 값 -- leakage 있음]', flush=True)
    print(f'  총 {(time.time() - t_start) / 60:.1f}분')
    print(f'  saved {out_path}')
    print(f'  saved {out_dir}/  '
          '(meta.json metrics.json metrics.csv best_model.pt '
          'test_acc_curve.png loss_curve.png time_breakdown.png)')


if __name__ == '__main__':
    main()
