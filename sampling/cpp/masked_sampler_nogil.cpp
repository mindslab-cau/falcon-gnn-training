// FALCON 마스킹 이웃 샘플러.
//
// 그래프를 다시 쓰지 않고, CSC 행을 훑을 때 후보를 걸러내는 방식으로 서브샘플링을 표현한다.
// 서로 독립적인 두 필터를 받고, 빈 텐서를 넘기면 그 필터는 꺼진다(오버헤드 0).
//
//   node_mask  bool[N]   이번 epoch에 살아남은 노드. False인 이웃은 후보에서 빠진다.
//                        -> 결과는 '살아남은 노드 집합에 유도된 부분그래프'의 행과 같다.
//   part_id    int32[N]  노드별 클러스터 id. 중심과 다른 클러스터인 이웃은 후보에서 빠진다.
//                        -> 결과는 'inter-cluster 간선을 모두 지운 그래프'의 행과 같다.
//   keep_inter bool[N]   'keep-inter-too' 예외 마스크. True인 노드는 part_id 필터를
//                        건너뛰어 inter-cluster 이웃까지 살린다. intra-only가 저차수
//                        노드를 고립시키는 것을 막는 heuristic (하위 10% degree train 노드).
//                        part_id가 없으면 아무 효과도 없다(끌 것이 없다).
//
//                        [Q3=a, 현재] 중심 기준: keep_inter[중심 v]가 True면 v의 모든
//                        inter 이웃을 살린다. 아래 필터 루프의 'bypass_part' 참고.
//   inter_keep_frac double  [R] inter 이웃 무작위 보존. 0이면 끔(기존과 동일). p이면 각
//                        inter 간선 (v,c)가 확률 p로 살아남는다 -- 즉 중심마다 자기 inter
//                        이웃 중 기댓값 p 비율이 후보에 합류한다. keep_inter(노드 전부/전무)
//                        와 직교하며 OR로 합쳐진다: bypass_part면 어차피 전부 산다.
//                        동전은 (v,c,inter_keep_seed) 해시라 무상태다 -- inter_coin 주석.
//   inter_keep_seed int64  위 동전의 씨앗. 호출자가 epoch 마다 바꿔 넘기는 것을 전제로 한다:
//                        한 epoch 안에서는 배치가 달라도 같은 inter 간선이 살고, epoch 이
//                        넘어가면 부분집합이 새로 뽑힌다. 배치 seed와 분리돼 있다.
//                        [Q3=b, 나중] 이웃 기준으로 바꾸려면: bypass_part 를 없애고
//                        neighbor 검사를 `part[c] != center_part && !keep_inter[c]` 로.
//
//   protect_inter bool[N] 이웃 기준 inter 예외 마스크. True인 이웃은 중심과
//                        클러스터가 달라도 후보로 살린다. node_mask는 유지된다.
//                        즉 저차수 '이웃'으로 가는 inter 간선만 살린다.
//
// part_id가 node_mask처럼 bool 마스크가 될 수 없는 이유: 간선의 생사가 이웃 노드 하나가
// 아니라 (중심, 이웃) 쌍에 달려 있다. 같은 이웃이 자기 클러스터 중심에서는 살고 다른
// 클러스터 중심에서는 죽는다. 그래서 마스크가 아니라 클러스터 id 배열을 받는다.
// keep_inter 도 같은 이유로 '중심에 따라' 간선 생사가 갈리므로 G'(고정 CSC)로는 표현할
// 수 없고, 반드시 이 런타임 필터에서만 구현된다.
//
// 필터는 항상 fanout 추첨 '이전'에 적용된다. 살아남은 이웃이 충분하면 fanout을 그대로
// 채운다 -- 먼저 뽑고 나중에 거르는 방식과 달리 실효 fanout이 줄지 않는다.
//
// 출력은 PyG NeighborLoader와 같은 '병합 서브그래프'다: L홉 전체의 합집합 노드 집합
// n_id(앞 batch_size개가 seed)와 그 위의 edge_index.
//
// 여기에 더해 hop 경계 두 개를 같이 돌려준다 -- cum_nodes / cum_edges. 노드는 발견 순서로,
// 간선은 생성 순서로 append되므로 'hop <= h 인 노드/간선'은 언제나 배열의 앞쪽 연속 구간이다.
// 즉 이 누적값만 알면 레이어별 부분그래프(MFG)를 재인덱싱 없이 prefix 슬라이싱으로 얻는다.
// 이게 없으면 모델이 매 레이어마다 L홉 노드 전부를 계산하게 된다(papers 3층 기준 GEMM 18.7배).
// 쓰는 쪽: model/models.py SAGE.forward 의 cum_nodes/cum_edges 인자.
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// [0, n)에서 서로 다른 k개를 뽑는다 (Floyd's algorithm). n 크기 버퍼를 만들지 않으므로
// 차수가 수만인 허브 노드에서도 O(k)다.
static inline void sample_k_distinct(int64_t n, int64_t k, std::mt19937_64 &gen,
                                     std::unordered_set<int64_t> &perm,
                                     std::vector<int64_t> &out) {
  perm.clear();
  perm.reserve(k * 2);
  for (int64_t j = n - k; j < n; j++) {
    int64_t r = std::uniform_int_distribution<int64_t>(0, j)(gen);
    if (!perm.insert(r).second) perm.insert(j);
  }
  out.assign(perm.begin(), perm.end());
}

// splitmix64 finalizer. inter_keep 의 동전던지기용 -- gen(mt19937_64)을 쓰지 않는 것이
// 핵심이다. 아래 alive() 는 기각 샘플링에서 같은 이웃을 여러 번 판정할 수 있고, 폴백
// (need_full_scan)에서 행을 처음부터 다시 훑는다. gen 을 소비하면 그때마다 답이 달라져
// '무작위 순열의 앞에서부터 생존자를 취한 것과 같다'는 등가성이 깨진다. 해시는 (v,c,salt)
// 의 순수 함수라 몇 번을 물어도 같은 답을 준다.
static inline uint64_t mix64(uint64_t z) {
  z += 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

// RANGE / NODISCARD 는 컴파일 타임 상수다. 런타임 분기로 두면 alive() 람다가 커져 인라인이 깨지고,
// 두 기능을 '꺼도' 기본 경로가 느려진다(실측: 필터 켬 87.5 -> 93.3 ms/batch). 특수화하면 끈 경로는
// 기능을 넣기 전과 같은 코드가 된다.
template <typename col_t, bool RANGE, bool NODISCARD>
std::tuple<torch::Tensor, torch::Tensor, int64_t, torch::Tensor, torch::Tensor>
sample_merged_impl(const int64_t *rowptr, const col_t *col, const int64_t *seeds,
                   int64_t num_seeds, const std::vector<int64_t> &fanouts,
                   const bool *mask, const int32_t *part, const bool *keep_inter,
                   const bool *protect_inter, bool replace, uint64_t seed,
                   uint64_t inter_keep_thresh, uint64_t inter_keep_seed,
                   const int64_t *bounds) {
  std::mt19937_64 gen(seed);
  // 동전던지기 소금. 배치 seed(gen 쪽)와 분리된 별도 씨앗이다 -- 호출자가 epoch 마다
  // 바꿔 넘기면 '이번 epoch 에 살아남을 inter 부분집합'이 epoch 단위로 다시 뽑히고,
  // 한 epoch 안에서는 어느 배치에서 보든 같은 간선이 산다. 배치 seed 를 섞으면
  // 배치마다 부분집합이 달라져 epoch 단위 고정이 깨지므로 여기에 넣지 않는다.
  const uint64_t coin_salt = mix64(inter_keep_seed ^ 0xa24baed4963ee407ULL);

  std::vector<int64_t> n_ids;                       // 지역 id -> 전역 id
  std::unordered_map<int64_t, int64_t> n_id_map;    // 전역 id -> 지역 id
  n_ids.reserve(num_seeds * 8);
  n_id_map.reserve(num_seeds * 8);
  for (int64_t i = 0; i < num_seeds; i++) {
    if (n_id_map.insert({seeds[i], (int64_t)n_ids.size()}).second) n_ids.push_back(seeds[i]);
  }
  const int64_t batch_size = (int64_t)n_ids.size();

  std::vector<int64_t> src, dst;                    // edge_index (이웃 -> 중심)
  std::vector<int64_t> cand, picked, chosen;        // 재사용 버퍼 (chosen = 최종 이웃 전역 id)
  std::vector<int64_t> tried;                       // 기각 샘플링에서 이미 뽑아본 행 위치
  std::vector<uint8_t> seen;                        // 위 위치의 중복 판정용 (touched만 되돌린다)
  std::unordered_set<int64_t> perm;

  // hop 경계. cum_nodes[h] = hop<=h 인 노드 수, cum_edges[h] = dst가 hop<=h 인 간선 수.
  // hop h의 확장이 만드는 간선은 (hop h+1 이웃 -> hop h 중심)이므로, layer 루프가 한 바퀴
  // 끝난 시점의 n_ids.size()/src.size()가 그대로 다음 경계가 된다.
  std::vector<int64_t> cum_nodes, cum_edges;
  cum_nodes.push_back(batch_size);                  // hop 0 = seed

  // 한 홉 = 직전 홉에서 '새로 추가된' 노드들만 확장한다 (PyG와 같은 규약).
  int64_t begin = 0, end = batch_size;
  for (size_t layer = 0; layer < fanouts.size(); layer++) {
    const int64_t k = fanouts[layer];
    // 이번 hop 이 만들 수 있는 최대치를 미리 확보한다. 확장할 노드가 (end-begin)개이고
    // 각각 최대 k개를 뽑으므로 상한이 정확히 (end-begin)*k 다.
    //
    // 이게 없으면 unordered_map 이 로드팩터를 넘길 때마다 '전체 원소 재해싱'을 한다.
    // 초기 예약 num_seeds*8(=8,192)에서 최종 337,375(papers bs1024 실측)까지 커지려면
    // 리해시가 6회 일어나고 누적 재해싱 원소가 67만 개에 달한다. hop 단위로 미리 잡으면
    // 리해시가 hop당 1회(그것도 맵이 아직 작을 때)로 줄어 누적이 약 6.8만 개가 된다.
    // k<0(평가용 전체 이웃)은 상한을 알 수 없으므로 건너뛴다.
    if (k > 0) {
      const int64_t add = (end - begin) * k;
      n_ids.reserve((size_t)((int64_t)n_ids.size() + add));
      n_id_map.reserve((size_t)((int64_t)n_id_map.size() + add));
      src.reserve((size_t)((int64_t)src.size() + add));
      dst.reserve((size_t)((int64_t)dst.size() + add));
    }
    for (int64_t i = begin; i < end; i++) {
      const int64_t v = n_ids[i];
      const int64_t r0 = rowptr[v], r1 = rowptr[v + 1];
      const int64_t row_count = r1 - r0;
      if (row_count == 0) continue;
      const int32_t center_part = (part != nullptr) ? part[v] : 0;
      // keep-inter-too: 중심 v가 예외 노드면 part_id 필터를 통째로 건너뛴다 (Q3=a).
      // Q3=b(이웃 기준)로 바꿀 때는 이 줄을 지우고 아래 neighbor 검사에
      // `&& !(keep_inter != nullptr && keep_inter[c])` 를 붙인다.
      const bool bypass_part = (keep_inter != nullptr && keep_inter[v]);
      // [방법1 part-range] 파티션이 연속 node id 구간 [plo, phi) 일 때만 켠다(호출자가 보장).
      // 이웃마다 part_id[c](444MB 배열 무작위 조회)를 읽는 대신 구간 비교로 같은 판정을 한다.
      const int64_t plo = RANGE ? bounds[center_part] : 0;
      const int64_t phi = RANGE ? bounds[center_part + 1] : 0;

      // --- 후보 선별 + fanout 추첨 -> chosen (이웃의 전역 id) ---
      // 두 필터는 교집합이라 순서 무관. 한 번에 판정한다.
      const bool filtered = (mask != nullptr) || (part != nullptr);
      // inter-keep-frac: 이 중심의 inter 이웃 각각을 독립 동전으로 살린다(생존 확률 p).
      // 중심당 '정확히 p 비율'이 아니라 기댓값이 p 인 이항분포다 -- 정확한 개수를 맞추려면
      // 행 단위 카운터가 필요하고, 그러면 alive() 가 상태를 갖게 되어 위 기각 샘플링의
      // 등가성이 깨진다. 무상태 해시를 쓰는 이유가 그것이다.
      const auto inter_coin = [&](int64_t c) -> bool {
        return (mix64(mix64((uint64_t)v * 0x9e3779b97f4a7c15ULL ^ (uint64_t)c)
                      ^ coin_salt) >> 11) < inter_keep_thresh;
      };
      const auto alive = [&](int64_t c) -> bool {
        if constexpr (RANGE) {
          // 판정 값은 아래 원본 경로와 같다(조건들의 AND). 메모리를 안 만지는 검사를 먼저 한다:
          // 구간 비교 -> 동전(해시) -> 그래도 남은 이웃만 mask 를 읽는다. 값이 같으므로
          // 난수 소비도 같고 출력은 비트 단위로 동일하다.
          if (!bypass_part && (c < plo || c >= phi) &&
              !(protect_inter != nullptr && protect_inter[c]) &&
              !(inter_keep_thresh > 0 && inter_coin(c))) return false;
          if (mask != nullptr && !mask[c]) return false;
          return true;
        }
        if (mask != nullptr && !mask[c]) return false;
        if (part != nullptr && !bypass_part && part[c] != center_part &&
            !(protect_inter != nullptr && protect_inter[c]) &&
            !(inter_keep_thresh > 0 && inter_coin(c))) return false;
        return true;
      };
      chosen.clear();

      // [기각 샘플링] 행 전체를 훑지 않고, 무작위 위치에서 생존 이웃 k개를 찾을 때까지만 본다.
      //
      // 왜 필요한가: fanout이 10인데도 예전 코드는 행 전체를 훑었다. 이웃 샘플링은 고차수
      // 노드를 뽑을 확률이 높아 프론티어가 허브로 편향되는데, papers 실측으로 확장 노드의
      // 평균 차수가 104.8(그래프 전체 평균 17.4의 6배, hop1은 159)이었다. 즉 10개를 쓰려고
      // 평균 105개의 node_mask(111MB 배열)를 랜덤 조회하고 있었다 -- 배치당 628만 회.
      //
      // 왜 분포가 같은가: 무작위 순열의 앞에서부터 생존자만 취한 것과 같고, 그것은 생존
      // 집합에서 균등하게 k개를 뽑은 것과 같다. 생존율 p면 기대 조회가 k/p (p~0.45, k=10
      // -> 약 22회)로 줄어든다.
      //
      // 예산 안에 k개를 못 채우면(생존 이웃이 k개 미만이거나 운이 나쁠 때) 아래 전체 스캔으로
      // 폴백한다. 폴백 경로도 균등하므로 최종 분포는 그대로다.
      bool need_full_scan = true;
      if (filtered && !replace && k > 0 && row_count > 4 * k) {
        if ((int64_t)seen.size() < row_count) seen.resize(row_count, 0);
        tried.clear();
        // 예산은 실패 판정용일 뿐이다 -- k개를 채우면 즉시 멈추므로 기대 비용은 k/p 그대로다.
        // [방법2 rej-nodiscard] 실패해도 재검사가 없으므로 예산을 늘려도 손해가 없다 -- 통과율이
        // 낮은 허브 행이 전체 스캔 없이 끝난다.
        const int64_t budget = std::min<int64_t>(
            row_count, NODISCARD ? std::max<int64_t>(4 * k + 16, row_count / 2) : 4 * k + 16);
        std::uniform_int_distribution<int64_t> u(0, row_count - 1);
        for (int64_t t = 0; t < budget && (int64_t)chosen.size() < k; t++) {
          const int64_t j = u(gen);
          if (seen[j]) continue;                    // 비복원: 같은 위치는 한 번만
          seen[j] = 1;
          tried.push_back(j);
          const int64_t c = (int64_t)col[r0 + j];
          if (alive(c)) chosen.push_back(c);
        }
        if ((int64_t)chosen.size() == k) {
          need_full_scan = false;
        } else if constexpr (NODISCARD) {
          // [방법2] 버리지 않고 이어간다. tried 는 균등한 위치 부분집합이고 그 안의 생존자는 전부
          // chosen 에 있다(< k). 모자란 k-c 개를 '아직 안 본 위치'의 생존자에서 균등하게 뽑으면
          // 무작위 순열의 앞에서부터 생존자 k 개를 취한 것과 정확히 같은 분포다. 이미 본 위치는
          // 다시 검사하지 않는다. (원본과 분포는 같고 난수열만 달라진다.)
          cand.clear();
          for (int64_t j = 0; j < row_count; j++) {
            if (seen[j]) continue;
            const int64_t c = (int64_t)col[r0 + j];
            if (alive(c)) cand.push_back(c);
          }
          const int64_t need = k - (int64_t)chosen.size();
          const int64_t m2 = (int64_t)cand.size();
          if (m2 <= need) {
            for (const int64_t c : cand) chosen.push_back(c);
          } else {
            sample_k_distinct(m2, need, gen, perm, picked);
            for (const int64_t p : picked) chosen.push_back(cand[p]);
          }
          need_full_scan = false;
        } else {
          chosen.clear();                           // 실패 -> 전체 스캔으로 정확히 다시
        }
        for (const int64_t q : tried) seen[q] = 0;  // 건드린 곳만 되돌린다 (O(t))
      }

      if (need_full_scan) {
        int64_t m;
        if (filtered) {
          cand.clear();
          cand.reserve(row_count);
          for (int64_t j = r0; j < r1; j++) {
            const int64_t c = (int64_t)col[j];
            if (alive(c)) cand.push_back(c);
          }
          m = (int64_t)cand.size();
        } else {
          m = row_count;                            // 필터 없음 -> 원본 행을 그대로 (복사 0)
        }
        if (m == 0) continue;
        const auto at = [&](int64_t p) -> int64_t {
          return filtered ? cand[p] : (int64_t)col[r0 + p];
        };
        if (k < 0 || m <= k) {                      // k<0 = 전부 (평가용 full neighbor)
          for (int64_t j = 0; j < m; j++) chosen.push_back(at(j));
        } else if (replace) {
          std::uniform_int_distribution<int64_t> u(0, m - 1);
          for (int64_t j = 0; j < k; j++) chosen.push_back(at(u(gen)));
        } else {
          sample_k_distinct(m, k, gen, perm, picked);
          for (const int64_t p : picked) chosen.push_back(at(p));
        }
      }
      if (chosen.empty()) continue;

      for (const int64_t c : chosen) {
        auto it = n_id_map.find(c);
        int64_t local;
        if (it == n_id_map.end()) {
          local = (int64_t)n_ids.size();
          n_id_map.emplace(c, local);
          n_ids.push_back(c);
        } else {
          local = it->second;
        }
        src.push_back(local);                       // 이웃
        dst.push_back(i);                           // 중심
      }
    }
    begin = end;
    end = (int64_t)n_ids.size();
    cum_nodes.push_back(end);                       // hop<=layer+1 누적 노드
    cum_edges.push_back((int64_t)src.size());       // dst가 hop<=layer 인 누적 간선
    if (begin == end) break;                        // 더 확장할 노드가 없다
  }
  // 조기 break(확장할 노드가 없음)로 hop이 모자라면 마지막 값으로 채워 길이를 레이어 수에
  // 맞춘다. 그 레이어들은 '새로 늘어난 것이 없는' 절단이 되어 결과가 달라지지 않는다.
  while ((int64_t)cum_nodes.size() < (int64_t)fanouts.size() + 1)
    cum_nodes.push_back(cum_nodes.back());
  while (cum_edges.size() < fanouts.size())
    cum_edges.push_back(cum_edges.empty() ? 0 : cum_edges.back());

  const int64_t E = (int64_t)src.size();
  auto opts = torch::TensorOptions().dtype(torch::kInt64);
  auto edge_index = torch::empty({2, E}, opts);
  auto ei = edge_index.data_ptr<int64_t>();
  std::copy(src.begin(), src.end(), ei);
  std::copy(dst.begin(), dst.end(), ei + E);

  auto n_id = torch::empty({(int64_t)n_ids.size()}, opts);
  std::copy(n_ids.begin(), n_ids.end(), n_id.data_ptr<int64_t>());

  auto cn = torch::empty({(int64_t)cum_nodes.size()}, opts);
  std::copy(cum_nodes.begin(), cum_nodes.end(), cn.data_ptr<int64_t>());
  auto ce = torch::empty({(int64_t)cum_edges.size()}, opts);
  std::copy(cum_edges.begin(), cum_edges.end(), ce.data_ptr<int64_t>());

  return std::make_tuple(edge_index, n_id, batch_size, cn, ce);
}

std::tuple<torch::Tensor, torch::Tensor, int64_t, torch::Tensor, torch::Tensor>
sample_merged(torch::Tensor rowptr, torch::Tensor col, torch::Tensor seeds,
              std::vector<int64_t> fanouts, torch::Tensor node_mask, torch::Tensor part_id,
              torch::Tensor keep_inter, torch::Tensor protect_inter,
              bool replace, int64_t seed, double inter_keep_frac,
              int64_t inter_keep_seed, torch::Tensor part_bounds, bool rej_nodiscard) {
  TORCH_CHECK(rowptr.dtype() == torch::kInt64, "rowptr must be int64");
  TORCH_CHECK(seeds.dtype() == torch::kInt64, "seeds must be int64");
  TORCH_CHECK(rowptr.is_contiguous() && seeds.is_contiguous(), "rowptr/seeds must be contiguous");

  const int64_t N = rowptr.numel() - 1;
  // 빈 텐서 = 그 필터 끄기. 셋 다 비면 원본 그래프를 그대로 걷는다.
  const bool *mask = nullptr;
  if (node_mask.numel() > 0) {
    TORCH_CHECK(node_mask.dtype() == torch::kBool && node_mask.numel() == N,
                "node_mask must be bool[num_nodes]");
    mask = node_mask.data_ptr<bool>();
  }
  const int32_t *part = nullptr;
  if (part_id.numel() > 0) {
    TORCH_CHECK(part_id.dtype() == torch::kInt32 && part_id.numel() == N,
                "part_id must be int32[num_nodes]");
    part = part_id.data_ptr<int32_t>();
  }
  // keep_inter 는 part_id 필터의 예외 마스크라, part_id 없이 주면 조용히 무시된다.
  const bool *keep = nullptr;
  if (keep_inter.numel() > 0) {
    TORCH_CHECK(keep_inter.dtype() == torch::kBool && keep_inter.numel() == N,
                "keep_inter must be bool[num_nodes]");
    keep = keep_inter.data_ptr<bool>();
  }
  const bool *protect = nullptr;
  if (protect_inter.numel() > 0) {
    TORCH_CHECK(protect_inter.dtype() == torch::kBool && protect_inter.numel() == N,
                "protect_inter must be bool[num_nodes]");
    protect = protect_inter.data_ptr<bool>();
  }

  const auto *rp = rowptr.data_ptr<int64_t>();
  const auto *sd = seeds.data_ptr<int64_t>();
  const int64_t ns = seeds.numel();
  const uint64_t s = (uint64_t)seed;

  // 동전 임계값. 해시 상위 53비트와 비교하므로 2^53 스케일이다 -- 2^64 로 잡으면
  // frac=1.0 에서 오버플로가 난다. frac<=0 -> 0 (기능 끔, alive() 에서 단락된다),
  // frac>=1 -> 2^53 이라 (h>>11) < thresh 가 항상 참(= part 필터를 완전히 끈 것과 같다).
  TORCH_CHECK(inter_keep_frac <= 1.0, "inter_keep_frac must be <= 1");
  const uint64_t thresh =
      (inter_keep_frac <= 0.0) ? 0ULL
                              : (uint64_t)(inter_keep_frac * 9007199254740992.0);

  // G'의 indices는 int32, 원본 CSC는 int64 -- 변환 없이 둘 다 받는다.
  const uint64_t ks = (uint64_t)inter_keep_seed;
  // [방법1] part_bounds: int64[K+1], 파티션 p 의 node id 구간이 [b[p], b[p+1]). 빈 텐서 = 끔.
  // part_id 필터가 꺼져 있으면 의미가 없으므로 무시한다.
  const int64_t *bd = nullptr;
  if (part != nullptr && part_bounds.defined() && part_bounds.numel() > 0) {
    TORCH_CHECK(part_bounds.dtype() == torch::kInt64 && part_bounds.is_contiguous() &&
                    part_bounds.numel() >= 2,
                "part_bounds must be contiguous int64[K+1]");
    bd = part_bounds.data_ptr<int64_t>();
    TORCH_CHECK(bd[0] == 0 && bd[part_bounds.numel() - 1] == N,
                "part_bounds must cover [0, num_nodes)");
  }
  // (col dtype) x (RANGE) x (NODISCARD) 조합별 특수화로 보낸다.
#define FALCON_CALL(CT, R, D)                                                              \
  return sample_merged_impl<CT, R, D>(rp, col.data_ptr<CT>(), sd, ns, fanouts, mask, part, \
                                      keep, protect, replace, s, thresh, ks, bd)
#define FALCON_DISPATCH(CT)                                  \
  do {                                                    \
    if (bd != nullptr && rej_nodiscard) FALCON_CALL(CT, true, true);   \
    if (bd != nullptr) FALCON_CALL(CT, true, false);         \
    if (rej_nodiscard) FALCON_CALL(CT, false, true);         \
    FALCON_CALL(CT, false, false);                           \
  } while (0)
  if (col.dtype() == torch::kInt32) FALCON_DISPATCH(int32_t);
  TORCH_CHECK(col.dtype() == torch::kInt64, "col must be int32 or int64");
  FALCON_DISPATCH(int64_t);
#undef FALCON_DISPATCH
#undef FALCON_CALL
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sample_merged", &sample_merged,
        // [nogil] 샘플링 동안 GIL 을 놓는다. 이 함수는 파이썬 객체를 만지지 않고(읽기 전용
        // 텐서 포인터 + 지역 버퍼 + torch::empty) 전역 상태도 없어 안전하다. GIL 을 쥔 채
        // ~100ms 를 돌면 DataLoader 워커의 Queue feeder / fd 전달 스레드가 그동안 멈춰,
        // 완성된 배치가 main 에 늦게 도착한다 (papers 실측: epoch 27.3s -> 20.9s, 출력 비트 동일).
#ifndef FALCON_HOLD_GIL          // -DFALCON_HOLD_GIL 로 빌드하면 원본처럼 GIL 을 쥔 채 돈다 (--no-gil-release ablation)
        py::call_guard<py::gil_scoped_release>(),
#endif
        "masked neighbor sampling -> merged L-hop subgraph "
        "(edge_index, n_id, batch_size, cum_nodes, cum_edges). "
        "cum_nodes[h]=nodes with hop<=h (len L+1), cum_edges[h]=edges with dst hop<=h (len L); "
        "both are prefix boundaries -- slice x[:cum_nodes[k]] / edge_index[:, :cum_edges[k]] "
        "to get the per-layer MFG without any re-indexing.",
        py::arg("rowptr"), py::arg("col"), py::arg("seeds"), py::arg("fanouts"),
        py::arg("node_mask"), py::arg("part_id"), py::arg("keep_inter"),
        py::arg("protect_inter"),
        py::arg("replace"), py::arg("seed"), py::arg("inter_keep_frac") = 0.0,
        py::arg("inter_keep_seed") = 0,
        py::arg("part_bounds") = torch::empty({0}, torch::kInt64),
        py::arg("rej_nodiscard") = false);
}
