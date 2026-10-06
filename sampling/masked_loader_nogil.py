"""마스킹 이웃 샘플러의 Python 쪽 — C++ 확장 로드 + 배치 조립.  [STEP 4]

cpp/masked_sampler.cpp 를 JIT으로 빌드해서(최초 1회, 이후 캐시) 쓴다. 하는 일:
  1) seed 노드 배치를 C++ 샘플러에 넘겨 L홉 병합 서브그래프를 받고
  2) 그 노드들의 feature/label을 원본 memmap에서 gather해
  3) PyG Data 로 감싸 넘긴다 (모델 코드는 그대로 batch.x / batch.edge_index 를 쓴다)

feature를 미리 복사해 두지 않고 배치마다 gather하는 것이 마스킹 방식의 핵심이다.
num_workers>0이면 그 gather가 워커 프로세스에서 GPU 연산과 겹쳐 돈다.

같은 샘플러가 학습과 평가를 모두 처리한다 -- 필터를 빈 텐서로 넘기면 원본 전체 그래프를
그대로 걷는다(오버헤드 0). 평가가 별도 코드 경로를 갖지 않는다.
"""
import os
from time import perf_counter as _perf

import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data

_EXT = None


def _ext():
    """C++ 확장을 지연 로드한다 (최초 호출 때만 컴파일, 이후 ~/.cache/torch_extensions 재사용)."""
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cpp', 'masked_sampler_nogil.cpp')
        # FALCON_HOLD_GIL=1 (--no-gil-release): 같은 소스를 GIL 보유판으로 빌드한다. 이름이 달라 캐시가 섞이지 않는다.
        hold = os.environ.get('FALCON_HOLD_GIL', '0') == '1'
        _EXT = load(name='falcon_masked_sampler_nogil' + ('_gilheld' if hold else ''), sources=[src],
                    extra_cflags=['-O2'] + (['-DFALCON_HOLD_GIL'] if hold else []), verbose=False)
    return _EXT


class _SeedDataset(Dataset):
    """seed 노드 id 목록. DataLoader가 이걸 배치로 잘라 collate_fn에 넘긴다."""

    def __init__(self, seeds):
        self.seeds = seeds

    def __len__(self):
        return self.seeds.numel()

    def __getitem__(self, i):
        return self.seeds[i]


class MaskedNeighborLoader:
    """마스킹 필터를 건 fanout 이웃 샘플링 로더.

    rowptr/col   그래프 CSC. col은 int32(G')/int64(원본) 둘 다 된다.
    x, y         원본 전체의 feature/label (memmap 기반 텐서). 배치마다 인덱싱한다.
    node_mask    bool[N] 또는 None. 이번 epoch 생존 노드 (서브샘플).
    part_id      int32[N] 또는 None. 주면 inter-cluster 간선이 샘플링에서 빠진다.
    keep_inter   bool[N] 또는 None. keep-inter-too 예외 마스크. True인 노드는 part_id
                 필터를 건너뛰어 inter 간선까지 살린다 (sampling/keep_inter.py 참고).
                 part_id 없이 주면 효과 없음.
    protect_inter bool[N] 또는 None. True인 이웃은 다른 클러스터여도 part_id 필터의
                  예외로 허용한다. node_mask 조건은 그대로 적용된다.
    inter_keep_frac float. [R] 0이면 끔. K이면 각 inter 간선이 확률 K로 part_id 필터를
                  통과한다(중심마다 자기 inter 이웃의 기댓값 K 비율). keep_inter/
                  protect_inter 와 직교하며 OR로 합쳐진다. part_id 없이 주면 무시된다.
    inter_keep_seed int. 위 동전의 씨앗. epoch 마다 바꿔 넘기면 살아남는 inter 부분집합이
                  epoch 단위로 다시 뽑힌다(한 epoch 안에서는 모든 배치가 같은 집합을 본다).
    fanouts      hop당 이웃 수. -1이면 그 hop은 이웃 전부 (평가용 full neighbor).
    """

    def __init__(self, rowptr, col, seeds, fanouts, x, y, node_mask=None, part_id=None,
                 keep_inter=None, protect_inter=None, inter_keep_frac=0.0, inter_keep_seed=0,
                 batch_size=1024, shuffle=True, num_workers=0,
                 pin_memory=False, replace=False, seed=0, profile=False, persistent=False,
                 part_bounds=None, rej_nodiscard=False):
        # profile=True 면 배치마다 collate 내부를 쪼개 재서 Data 에 붙인다(t_sample/t_gather).
        # 평상시엔 끄는 이유: 배치당 perf_counter 4회는 무해하지만, 측정용 필드가 배치에
        # 섞이면 학습 경로에서 의도치 않게 참조될 수 있다.
        self.profile = bool(profile)
        self.rowptr, self.col, self.x, self.y = rowptr, col, x, y
        self.fanouts = [int(f) for f in fanouts]
        self.replace = bool(replace)
        self.epoch_seed = int(seed)
        # 빈 텐서 = 그 필터 끄기 (C++ 쪽 규약). None을 여기서 빈 텐서로 바꾼다.
        self.node_mask = torch.empty(0, dtype=torch.bool) if node_mask is None else node_mask
        self.part_id = torch.empty(0, dtype=torch.int32) if part_id is None else part_id
        self.keep_inter = torch.empty(0, dtype=torch.bool) if keep_inter is None else keep_inter
        self.protect_inter = (torch.empty(0, dtype=torch.bool) if protect_inter is None
                              else protect_inter)
        # [R] inter 이웃 무작위 보존 비율. 0 = 끔(기존과 동일). part_id 없이 주면 C++ 쪽
        # part 필터 자체가 없으므로 조용히 무시된다 -- keep_inter 와 같은 이유다.
        self.inter_keep_frac = float(inter_keep_frac)
        # 위 동전의 씨앗. 학습 로더는 매 epoch 새로 만들어지므로 호출자가 epoch 마다 다른
        # 값을 넘긴다 -> 한 epoch 안에서는 배치가 달라도 같은 inter 간선이 살고, epoch 이
        # 넘어가면 새로 뽑힌다. 배치 난수(_collate 의 s)와는 별개다.
        self.inter_keep_seed = int(inter_keep_seed)
        # [방법1/2] masked_loader_opt_nogil 과 같은 의미. late-gather 를 끈 경로도 같은 플래그를 받게 한다.
        self.part_bounds = (torch.empty(0, dtype=torch.int64) if part_bounds is None else part_bounds)
        self.rej_nodiscard = bool(rej_nodiscard)

        if seeds.dtype == torch.bool:                    # 마스크로 줘도 되게
            seeds = seeds.nonzero(as_tuple=False).view(-1)
        self.seeds = seeds.to(torch.int64)

        # 기본이 persistent_workers=False 인 이유: 학습 로더는 매 epoch 마스크가 바뀐다.
        # 워커를 살려두면 fork 시점의 마스크 사본을 계속 쓰므로 갱신이 반영되지 않는다
        # (조용히 틀린 학습). 매 epoch 새로 fork하면 그 시점의 마스크를 본다.
        #
        # persistent=True 는 '마스크가 끝까지 안 바뀌는' 로더 전용이다. 대표적으로 평가
        # 로더가 그렇다 -- node_mask/part_id 를 None 으로 만들어 원본 그래프를 그대로 걷고,
        # 이후 한 번도 바꾸지 않는다. 이때 워커를 살려두면 epoch마다 되풀이하던
        # 프로세스 재생성과 페이지 테이블 콜드 스타트를 지불하지 않는다.
        # (fresh 워커의 첫 배치는 이후 평균의 2배 이상 느리다: 8워커 경합 시 426 vs 175 ms)
        #
        # 잘못 쓰면 조용히 틀리는 종류의 옵션이라, 아래 두 setter 가 persistent 로더에서
        # 호출되면 예외를 던진다. num_workers=0 이면 torch 가 애초에 이 조합을 거부한다.
        self.persistent = bool(persistent) and num_workers > 0
        self.loader = DataLoader(
            _SeedDataset(self.seeds), batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory, collate_fn=self._collate,
            persistent_workers=self.persistent)

    def _reject_if_persistent(self, what):
        if self.persistent:
            raise RuntimeError(
                f'persistent=True 로더에서는 {what} 를 바꿀 수 없다. 워커가 살아 있어 '
                f'갱신이 반영되지 않고 조용히 틀린 결과가 나온다. 마스크가 바뀌는 로더는 '
                f'persistent=False 로 만들 것.')

    def set_node_mask(self, node_mask):
        """이번 epoch의 생존 마스크로 교체. None이면 필터 없음(원본 그래프 전체)."""
        self._reject_if_persistent('node_mask')
        self.node_mask = torch.empty(0, dtype=torch.bool) if node_mask is None else node_mask

    def set_keep_inter_mask(self, keep_inter):
        """keep-inter-too 예외 마스크로 교체. None이면 예외 없음(순수 intra-only).

        Q1=a(원본 degree 고정)에서는 시작 시 한 번만 부르면 되지만, Q1=b(매 epoch
        subsample degree 재계산, keep_inter.py 참고)로 확장할 때 이 setter 를 매 epoch
        호출한다 -- set_node_mask 와 같은 이유로 persistent_workers=False 라, 다음 epoch
        의 fork 가 갱신된 마스크를 그대로 본다.
        """
        self._reject_if_persistent('keep_inter')
        self.keep_inter = torch.empty(0, dtype=torch.bool) if keep_inter is None else keep_inter

    def _collate(self, seed_list):
        seeds = torch.stack(seed_list) if isinstance(seed_list[0], torch.Tensor) \
            else torch.tensor(seed_list, dtype=torch.int64)
        # 배치마다 다른 난수. torch 전역 RNG에서 뽑으므로 torch.manual_seed로 재현된다.
        s = int(torch.randint(0, 2 ** 62, (1,)).item())
        t0 = _perf() if self.profile else 0.0
        edge_index, n_id, bs, cum_nodes, cum_edges = _ext().sample_merged(
            self.rowptr, self.col, seeds, self.fanouts,
            self.node_mask, self.part_id, self.keep_inter, self.protect_inter,
            self.replace, s, self.inter_keep_frac, self.inter_keep_seed,
            self.part_bounds, self.rej_nodiscard)
        t1 = _perf() if self.profile else 0.0

        # 여기가 memmap 페이지 폴트가 나는 지점. 워커 프로세스에서 돌면 GPU와 겹친다.
        data = Data(x=self.x[n_id], edge_index=edge_index, y=self.y[n_id[:bs]])
        data.n_id = n_id
        data.batch_size = int(bs)
        # hop 경계는 '파이썬 리스트'로 붙인다. 텐서로 두면 batch.to(device)가 GPU로 옮기고,
        # 그러면 레이어마다 슬라이스 인덱스를 읽을 때 D2H 동기화가 걸린다(배치당 6회).
        data.cum_nodes = cum_nodes.tolist()
        data.cum_edges = cum_edges.tolist()
        if self.profile:
            # 워커 안에서 측정한 '작업량'. 메인 루프의 data(대기 시간)와 다른 값이다 --
            # 워커가 병렬로 돌아 GPU와 겹치므로, 작업량 합계 > 대기 시간이 정상이다.
            data.t_sample = t1 - t0          # C++ 이웃 샘플링 (그래프 스캔 + 필터)
            data.t_gather = _perf() - t1     # feature/label gather (원본 memmap 읽기)
            data.t_sample_start = t0
            data.t_sample_end = t1
        return data

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)
