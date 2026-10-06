"""keep-inter-too: intra-only 대신 저차수 노드의 inter-cluster 간선을 살리는 필터.  [STEP 1.5]

동기
  intra-only(inter 간선 전부 삭제)는 저차수 노드를 고립시킨다 -- 이웃이 몇 개 없는데
  그마저 다른 클러스터에 있으면 그 노드는 이웃 0이 된다. keep-inter-too는 '가장 취약한'
  노드에 한해 inter 간선을 살려 이 고립을 막고, 고차수 노드는 그대로 intra만 유지해
  간선 축소 이득을 지킨다.

산출물
  keep_inter_mask : bool[N]. True인 노드는 C++ 샘플러(masked_sampler.cpp)에서 part_id
  필터를 건너뛴다. 이 마스크가 '누구를 True로 하느냐'가 방법의 전부다.

현재 구현과 확장 여지
  마스크를 채우는 방식이 두 축(Q1 degree 시점, Q2/Q4 모집단)으로, 필터를 거는 기준이
  한 축(Q3 중심/이웃)으로 갈린다. Q1 은 a/b 둘 다 구현했고, Q2/Q4=b 와 Q3=b 는 아직
  서명과 주석만 둔다.

    Q1  degree 를 어느 그래프에서 재나
        a  원본 전체 그래프 degree, 시작 시 1회 고정   -> compute_keep_inter_mask_static
        b  매 epoch subsample 된 그래프 degree, 재계산  -> compute_keep_inter_mask_subsampled
           (구현됨) 원본 degree 가 높아도 이웃이 죄다 탈락하면 하위로 내려오므로,
           '취약'을 그래프 고정 속성이 아니라 이번 epoch 의 상태로 판정한다. rtintra 처럼
           part_id 를 샘플링 때 거는 경로가 전제다 -- 세는 degree 의 정의가 샘플러의
           alive() 와 같아야 의미가 있기 때문이다.
    Q2/Q4  하위 frac 의 모집단
        a  train 노드 중                                -> population='train' (구현됨)
        b  전체 노드 중 (모든 hop 중심에 적용)          -> population='all'  (스텁)
    Q3  중심 v 기준 / 이웃 c 기준
        a  중심                                         -> masked_sampler.cpp 의 bypass_part
        b  이웃                                         -> 같은 파일 주석 참고 (C++ 한 줄)
"""
import numpy as np
import torch


def _degree_from_rowptr(rowptr) -> np.ndarray:
    """CSC indptr 에서 노드별 degree. 원본 대칭 CSC 기준이라 in==out."""
    rp = rowptr.numpy() if isinstance(rowptr, torch.Tensor) else np.asarray(rowptr)
    return (rp[1:] - rp[:-1]).astype(np.int64)


def _bottom_frac_threshold(deg: np.ndarray, population_idx: np.ndarray, frac: float) -> float:
    """population 노드들의 degree 중 하위 frac 경계값. 그 이하(<=)가 예외 대상이 된다."""
    if population_idx.size == 0:
        return -1.0
    vals = deg[population_idx]
    # 하위 frac 분위수. 동률이 경계에 몰리면 실제 선택 비율이 frac 을 넘을 수 있으나,
    # '저차수 보호'라는 목적상 경계 동률을 함께 살리는 쪽이 안전하다.
    return float(np.quantile(vals, frac, method='lower'))


def compute_keep_inter_mask_static(rowptr, train_mask, frac=0.1, verbose=True) -> torch.Tensor:
    """[Q1=a, Q2=a] 원본 degree 기준, train 노드 중 하위 frac 을 예외로 하는 고정 마스크.

    rowptr      원본 CSC indptr (int64[N+1]). degree = rp[v+1]-rp[v].
    train_mask  bool[N]. 하위 frac 을 고르는 모집단(= target 노드).
    frac        예외로 살릴 하위 비율 (기본 0.1 = 하위 10%).

    반환: bool[N]. (train 노드) AND (원본 degree <= train 하위 frac 경계). 나머지 False.
    이 함수는 시작 시 1회만 부르고 결과를 고정해 쓴다(원본 degree는 epoch에 무관하므로).
    """
    tm = train_mask.numpy() if isinstance(train_mask, torch.Tensor) else np.asarray(train_mask)
    deg = _degree_from_rowptr(rowptr)
    train_idx = np.nonzero(tm)[0]
    thr = _bottom_frac_threshold(deg, train_idx, frac)

    mask = np.zeros(deg.shape[0], dtype=bool)
    mask[train_idx] = deg[train_idx] <= thr
    out = torch.from_numpy(mask)
    if verbose:
        n_train = int(train_idx.size)
        n_keep = int(mask.sum())
        print(f'[keep-inter] Q1=a Q2=a · train {n_train:,} 중 degree<={thr:.0f} '
              f'하위 {frac:.0%} -> 예외 {n_keep:,}개 ({n_keep / max(n_train, 1):.1%}) '
              f'가 inter 간선 유지', flush=True)
    return out


# ---------------------------------------------------------------------------
# b 변형
# ---------------------------------------------------------------------------
def _epoch_degree(rowptr, col, keep_mask, nodes, part=None, block_edges=50_000_000):
    """nodes 각각의 '이번 epoch 유효 degree'를 센다.

    유효 = 서브샘플 생존(keep_mask) 이고, part 를 주면 중심과 같은 클러스터(intra)인 이웃.
    이것이 masked_sampler 의 alive() 가 bypass 없이 셀 후보 수와 정확히 같은 정의다 --
    그래야 '이 중심은 이번 epoch 에 이웃이 몇 개인가'를 재는 것이 된다.

    행을 블록으로 모아 한 번에 훑는다(materialize 와 같은 방식). 노드 수가 아니라 간선
    수로 블록을 자르는 이유: 고차수 노드가 몰린 구간에서 임시 배열이 터지지 않게 하려고.
    """
    rp = rowptr.numpy() if isinstance(rowptr, torch.Tensor) else np.asarray(rowptr)
    n = int(nodes.size)
    deg = np.zeros(n, dtype=np.int64)
    if n == 0:
        return deg
    starts_all = rp[nodes]
    seg_all = rp[nodes + 1] - starts_all
    # 간선 누적으로 블록 경계를 잡는다. append(n) 으로 마지막 경계를 못박아, 총 간선이
    # block_edges 보다 적어도(=경계가 하나도 안 생겨도) 전체가 한 블록으로 처리된다.
    cum = np.cumsum(seg_all)
    marks = np.arange(block_edges, int(cum[-1]) + 1, block_edges)
    bounds = np.unique(np.clip(np.append(np.searchsorted(cum, marks) + 1, n), 1, n))

    j0 = 0
    for j1 in bounds:
        j1 = int(j1)
        blk = nodes[j0:j1]
        starts = starts_all[j0:j1]
        seg = seg_all[j0:j1]
        total = int(seg.sum())
        if total == 0:
            deg[j0:j1] = 0
            j0 = j1
            continue
        # 행마다 [starts[k], starts[k]+seg[k]) 를 이어붙인 평탄 인덱스
        row_local = np.repeat(np.arange(j1 - j0, dtype=np.int64), seg)
        run_base = np.repeat(np.cumsum(seg) - seg, seg)
        src_index = np.repeat(starts, seg) + (np.arange(total, dtype=np.int64) - run_base)
        cols = np.asarray(col[src_index]).astype(np.int64, copy=False)

        alive = keep_mask[cols]
        if part is not None:
            alive &= part[cols] == part[blk][row_local]
        deg[j0:j1] = np.bincount(row_local[alive], minlength=(j1 - j0))
        j0 = j1
    return deg


def compute_keep_inter_mask_subsampled(rowptr, col, keep_mask, train_mask,
                                       part_id=None, frac=0.1, verbose=True) -> torch.Tensor:
    """[Q1=b] 이번 epoch subsample 된 그래프에서의 degree 로 하위 frac 을 다시 고른다.

    static 버전과 달리 매 epoch 부른다: keep_mask 로 죽은 이웃을 뺀 뒤(그리고 part_id 를
    주면 inter 이웃까지 뺀 뒤) 남은 이웃 수가 '이번 epoch 의 degree'다. subsample 으로
    이웃을 잃어 이번 epoch 새로 취약해진 노드를 매번 보호하게 된다.

    static 과 무엇이 다른가: 원본 degree 는 epoch 에 무관하므로 static 은 늘 같은 노드
    집합을 고른다. 여기서는 원본 degree 가 높아도 이웃이 죄다 탈락했으면 하위로 내려오고,
    반대로 원본 degree 가 낮아도 이웃이 다 살아남았으면 구제 대상에서 빠진다. 즉 '취약'을
    그래프 고정 속성이 아니라 이번 epoch 의 상태로 판정한다.

    rowptr/col  원본 CSC (inter 포함). part_id 를 같이 줘야 intra 기준으로 센다.
    keep_mask   bool[N]. 이번 epoch Subsample.keep_mask.
    train_mask  bool[N]. 하위 frac 을 고르는 모집단(= target 노드).

    반환: bool[N]. compute_keep_inter_mask_static 과 같은 형식이다.
    학습 로더를 매 epoch 새로 만드는 구조면 생성 인자로 넘기면 되고, 로더를 재사용하는
    구조면 loader.set_keep_inter_mask(...) 로 갈아끼운다(persistent=False 여야 한다).
    """
    tm = train_mask.numpy() if isinstance(train_mask, torch.Tensor) else np.asarray(train_mask)
    km = keep_mask.numpy() if isinstance(keep_mask, torch.Tensor) else np.asarray(keep_mask)
    part = None
    if part_id is not None:
        part = part_id.numpy() if isinstance(part_id, torch.Tensor) else np.asarray(part_id)

    train_idx = np.nonzero(tm)[0]
    deg = _epoch_degree(rowptr, col, km, train_idx, part=part)
    # 모집단이 train 이므로 population_idx 는 deg 자신의 인덱스다(deg 가 이미 train 전용).
    thr = _bottom_frac_threshold(deg, np.arange(deg.size), frac)

    mask = np.zeros(tm.shape[0], dtype=bool)
    mask[train_idx] = deg <= thr
    out = torch.from_numpy(mask)
    if verbose:
        n_keep = int(mask.sum())
        n_iso = int((deg == 0).sum())
        print(f'[keep-inter] Q1=b Q2=a · train {train_idx.size:,} 중 epoch degree<={thr:.0f} '
              f'하위 {frac:.0%} -> 예외 {n_keep:,}개 ({n_keep / max(train_idx.size, 1):.1%}) '
              f'가 inter 간선 유지 (이웃 0인 노드 {n_iso:,}개)', flush=True)
    return out


def compute_keep_inter_mask(rowptr, train_mask, frac=0.1, population='train',
                            degree='static', col=None, keep_mask=None, part_id=None,
                            verbose=True) -> torch.Tensor:
    """[Q1/Q2/Q4] degree 시점 + 모집단 선택 디스패처.

    degree='static'     [Q1=a] 원본 degree, 시작 시 1회 고정. rowptr 만 있으면 된다.
    degree='subsampled' [Q1=b] 이번 epoch 유효 degree, 매 epoch 재계산.
                        col/keep_mask 가 필수이고, part_id 를 줘야 intra 기준으로 센다.

    population='train' : train 노드 중 하위 frac (현재 기본).
    population='all'   : [Q4=b] 전체 노드 중 하위 frac. 이러면 예외가 train seed 뿐 아니라
                         모든 hop 의 중심에 적용된다(저차수 중심이면 누구든 inter 유지).
                         구현 시 _bottom_frac_threshold 의 population_idx 를 전체 노드로.
    """
    if population == 'all':
        raise NotImplementedError('Q4=b (population=all) 미구현 -- 서명/주석만')
    if population != 'train':
        raise ValueError(f'unknown population {population!r}')
    if degree == 'static':
        return compute_keep_inter_mask_static(rowptr, train_mask, frac, verbose)
    if degree == 'subsampled':
        if col is None or keep_mask is None:
            raise ValueError("degree='subsampled' 에는 col 과 keep_mask 가 필요하다")
        return compute_keep_inter_mask_subsampled(rowptr, col, keep_mask, train_mask,
                                                  part_id=part_id, frac=frac, verbose=verbose)
    raise ValueError(f'unknown degree {degree!r}')
