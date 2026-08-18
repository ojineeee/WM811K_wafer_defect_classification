# 기술 상세 리포트

> 이 문서는 개발자/데이터 분석 담당자를 위한 상세 기술 문서입니다.
> 프로젝트 배경과 요약은 [메인 README](../README.md)를 먼저 봐주세요.

# WM-811K 웨이퍼맵 결함 패턴 분류

## 데이터

- 출처: [MIR Lab WM-811K](http://mirlab.org/dataset/public/MIR-WM811K.zip) (로그인 불필요, 344MB zip)
- 원본 pkl: `data/raw/extracted/MIR-WM811K/Python/WM811K.pkl` — pandas DataFrame, 컬럼 `[dieSize, failureType, lotName, trainTestLabel, waferIndex, waferMap]`
- 전체 811,457행 중 `failureType`이 실제 라벨(9개 클래스 중 하나)인 행은 172,950개(21.31%). 나머지는 라벨 없음(원본에 문자열 `'0'`으로 채워져 있어 `data.py`에서 명시적으로 필터링).

### 클래스 분포 (라벨 있는 subset 기준)

| 클래스 | 개수 | 비율 |
|---|---|---|
| none | 147,431 | 85.24% |
| Edge-Ring | 9,680 | 5.60% |
| Edge-Loc | 5,189 | 3.00% |
| Center | 4,294 | 2.48% |
| Loc | 3,593 | 2.08% |
| Scratch | 1,193 | 0.69% |
| Random | 866 | 0.50% |
| Donut | 555 | 0.32% |
| Near-full | 149 | 0.09% |

none 대 최소 클래스(Near-full) 비율: **약 990:1**.

### 웨이퍼맵 크기

- 세로: 15~212, 가로: 3~204
- 서로 다른 (height, width) 조합 346가지

## 전처리 파이프라인 (`src/data.py`, `src/train_cnn.py`)

1. `load_labeled()` — `failureType`이 9개 클래스 중 하나인 행만 필터링
2. `build_balanced_subset(df, cap=2000)` — 클래스별로 최대 2,000장까지만 샘플링 (그보다 적은 클래스는 전량 사용)
   - 최종 subset: none 2,000 / Edge-Ring 2,000 / Edge-Loc 2,000 / Center 2,000 / Loc 2,000 / Scratch 1,193 / Random 866 / Donut 555 / Near-full 149 = **12,763장**
3. `resize_wafer()` — `PIL.Image.resize(..., resample=Image.NEAREST)`로 64×64 통일 (범주형 값 0/1/2 보존 목적으로 최근접 이웃 사용). 0,1,2 값은 0/0.5/1.0으로 정규화.
4. `train_test_split(test_size=0.2, stratify=labels, random_state=42)`

## 모델

```python
SmallCNN(
  Conv2d(1, 16, 3) -> ReLU -> MaxPool2d(2)
  Conv2d(16, 32, 3) -> ReLU -> MaxPool2d(2)
  Conv2d(32, 64, 3) -> ReLU -> AdaptiveAvgPool2d(4)
  Flatten -> Linear(1024, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 9)
)
```

- Optimizer: Adam(lr=1e-3)
- Loss: `CrossEntropyLoss` with per-class weight `1/sqrt(class_count)` (서브샘플 이후에도 남은 불균형 보정)
- Epochs: 15, Batch size: 64
- Device: CPU (4 cores) — 전처리+학습 총 소요시간 약 **224초**

## 최종 성능 (held-out test, n=2,553)

| 지표 | 값 |
|---|---|
| Accuracy | 0.8014 |
| Macro F1 | 0.7984 |
| Weighted F1 | 0.7932 |

### 클래스별 F1

| 클래스 | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| none | 0.83 | 0.81 | 0.82 | 400 |
| Center | 0.91 | 0.94 | 0.93 | 400 |
| Donut | 0.83 | 0.94 | 0.88 | 111 |
| Edge-Loc | 0.74 | 0.83 | 0.78 | 400 |
| Edge-Ring | 0.94 | 0.95 | 0.95 | 400 |
| Loc | 0.66 | 0.66 | 0.66 | 400 |
| Random | 0.86 | 0.96 | 0.90 | 173 |
| Scratch | 0.47 | 0.31 | 0.37 | 239 |
| Near-full | 0.96 | 0.83 | 0.89 | 30 |

### Scratch 오분류 breakdown

실제 Scratch(239건) 중 예측 결과 분포 (같은 설정으로 재학습한 별도 검증 실행 기준):

| 예측된 클래스 | 건수 |
|---|---|
| none | 73 |
| Loc | 64 |
| Scratch (정답) | 69 |
| Edge-Loc | 25 |
| 기타 | 8 |

## 파생변수 실험 (`src/derived_features.py`)

### 동기

Scratch의 저조한 성능을 분석하던 중, 클래스별 불량 밀도에서 다음을 발견:

| 클래스 | defect_density (평균) |
|---|---|
| Near-full | 0.877 |
| Random | 0.481 |
| Donut | 0.277 |
| Center | 0.231 |
| Edge-Loc | 0.184 |
| Edge-Ring | 0.151 |
| Loc | 0.148 |
| none | 0.104 |
| **Scratch** | **0.101** |

**Scratch의 불량 밀도가 none보다도 낮다.** 얇은 선형 결함이라 불량 다이 수 자체가 적고, 64×64 nearest 리사이즈에서 선이 끊기거나 소실되어 CNN이 none과 구분하지 못하는 것으로 판단.

### 설계한 파생변수 (8개)

원본 해상도(리사이즈 전) 웨이퍼맵에서 계산:

| 변수 | 정의 | 의도 |
|---|---|---|
| `defect_density` | 불량 다이 / 유효 다이 | 전역 심각도 |
| `defect_radial_mean` | 불량 좌표의 중심으로부터 정규화 거리 평균 | Center vs Edge 계열 구분 |
| `defect_radial_std` | 위 거리의 표준편차 | Ring(균일) vs Loc(집중) 구분 |
| `defect_spread` | 불량 좌표 (y,x) 표준편차의 norm | Scratch(퍼짐) vs Loc(뭉침) |
| `defect_row_concentration` | 한 행에 몰린 불량 최대 비율 | 선형 패턴 포착 |
| `defect_col_concentration` | 한 열에 몰린 불량 최대 비율 | 선형 패턴 포착 |
| `wafer_area` | 유효 다이 총 개수 | 웨이퍼 규격 차이 |
| `aspect_ratio` | width / height | 웨이퍼 규격 차이 |

`StandardScaler`는 train split으로만 fit해 누수를 방지.

### 아키텍처

`HybridCNN` — `SmallCNN`과 동일한 conv 백본을 쓰되, flatten 출력(1024차원)에 파생변수 8개를 concat하여 분류기에 입력. `use_feats=False`로 두면 정확히 baseline과 동일한 구조가 되어, **파생변수 유무만 격리해서 비교**할 수 있음.

### 결과 (동일 split·동일 시드)

| 지표 | CNN only | CNN + derived | Δ |
|---|---|---|---|
| Macro F1 | 0.7902 | **0.8436** | +0.0534 |
| Accuracy | 0.8022 | **0.8347** | +0.0325 |

클래스별 F1:

| 클래스 | CNN only | CNN + derived | Δ |
|---|---|---|---|
| **Scratch** | 0.3690 | **0.5961** | **+0.2271** |
| Near-full | 0.8108 | 0.9677 | +0.1569 |
| Edge-Loc | 0.7788 | 0.8285 | +0.0497 |
| Loc | 0.6711 | 0.7077 | +0.0366 |
| none | 0.7986 | 0.8256 | +0.0270 |
| Edge-Ring | 0.9550 | 0.9598 | +0.0048 |
| Center | 0.9320 | 0.9338 | +0.0018 |
| Random | 0.8994 | 0.8987 | -0.0007 |
| Donut | 0.8966 | 0.8745 | -0.0221 |

원 가설(Scratch의 선형성·저밀도를 파생변수가 보완)과 정확히 일치하는 방향으로 개선됨. Donut만 소폭 하락.

## 데이터 증강 실험 (`src/augmentation.py`)

### 방법

- 증강 대상: 학습 샘플이 적은 5개 클래스 (`Scratch`, `Random`, `Donut`, `Near-full`, `Loc`)
- 변형: 90도 단위 회전 4종 × 좌우반전 2종 = 최대 8배
- 전제: 이 증강은 "결함 유형 분류"만을 목표로 한다. 웨이퍼의 방향(회전 상태) 자체가 공정
  진단에 의미를 갖는 과제(예: 장비 방향과 결함 위치의 상관관계 분석)에는 방향 정보를 지우는
  이 증강을 그대로 적용하면 안 된다.
- **누수 방지**: `train_test_split`을 먼저 실행해 test 인덱스를 고정한 뒤, train 인덱스에 대해서만 증강 적용. test에는 원본만 사용.
- 그 외 파이프라인(리사이즈, 파생변수, `HybridCNN`, 클래스 가중치)은 4단계와 동일 — **증강 유무만 격리해서 비교**.

### 결과 (동일 split·동일 시드, 이번 실행 기준)

| 지표 | 증강 없음 (train=10,210) | 증강 적용 (train=36,880) | Δ |
|---|---|---|---|
| Macro F1 | 0.8396 | **0.8815** | +0.0419 |
| Accuracy | 0.8327 | **0.8664** | +0.0337 |

클래스별 F1 변화:

| 클래스 | 증강 없음 | 증강 적용 | Δ |
|---|---|---|---|
| **Scratch** | 0.6087 | **0.7500** | **+0.1413** |
| Loc | 0.6804 | 0.7683 | +0.0879 |
| Donut | 0.8776 | 0.9474 | +0.0698 |
| Random | 0.8877 | 0.9464 | +0.0587 |
| Center | 0.9236 | 0.9444 | +0.0208 |
| Edge-Ring | 0.9544 | 0.9571 | +0.0027 |
| none | 0.8606 | 0.8620 | +0.0014 |
| Near-full | 0.9492 | 0.9492 | 0.0000 |
| Edge-Loc | 0.8140 | 0.8089 | -0.0051 |

증강을 적용한 5개 클래스가 전부 개선되었고(Near-full은 이미 F1 0.95로 포화 상태라 변화 없음), 증강하지 않은 Center/Edge-Ring/none/Edge-Loc은 거의 변화가 없거나 소폭 하락에 그침 — 증강 효과가 대상 클래스에 국한되어 나타나는, 기대한 그대로의 패턴.

주의: 이 실행의 "증강 없음" macro F1(0.8396)은 4단계에서 보고한 0.8436과 소폭 다른데, 이는 별도 프로세스 재실행에 따른 PyTorch CPU 연산의 미세한 비결정성 때문이며 두 수치 모두 유효한 반복 실행 결과입니다. (이후 `train_cnn.py`, `derived_features.py`, `augmentation.py`에 `torch.set_num_threads(1)`을 적용해 앞으로의 실행부터는 이 문제가 재발하지 않도록 고쳤습니다.)

## lot 순서 기반 drift 분석 및 재검증 (`src/lot_drift.py`, `src/lot_split_validation.py`)

### 동기

지금까지의 모든 실험은 무작위 분할 기준이었다. 제조 데이터는 시간/생산순서에 따른 드리프트가
흔하다는 점에서 이 위험을 점검할 필요가 있었다. WM-811K에는 타임스탬프가 없지만 `lotName`이
`lot{N}` 형식의 순번이라 생산 순서의 proxy로 사용했다.

### drift 분석 방법 및 결과

`lot_num`(정수 추출) 기준으로 라벨된 172,950건을 정렬 후 10분위로 나눠 불량 비율(`failureType
!= 'none'`)을 계산했다.

| 분위 | 불량 비율 |
|---|---|
| 0 | 0.4740 |
| 1 | 0.5402 |
| 2 | 0.0611 |
| 3 | 0.0490 |
| 4 | 0.0346 |
| 5 | 0.0591 |
| 6 | 0.0591 |
| 7 | 0.1006 |
| 8 | 0.0361 |
| 9 | 0.0616 |

초반 2개 분위(전체의 20%) 평균 0.5071 vs 나머지 8개 분위 평균 0.0576 — **약 8.8배 차이**로
매우 극단적인 드리프트다. lot 순서 자체가 균등 간격의 시간이 아니라는 점(초반 lot일수록 결함
비율이 높다는 건 라벨링 정책 혹은 실제 초기 공정 불안정성 둘 다 가능한 원인이며, 메타데이터만
으로는 구분 불가)에 유의해야 한다.

### 버그 수정 이력

최초 구현은 `df.sort_values('lot_num')` 후 **행(웨이퍼) 개수의 80%** 를 split 경계로 잘랐다.
lot 하나에 웨이퍼가 여러 장 속해 있을 수 있으므로, 이 방식은 "고유 lot 개수"가 아니라
"웨이퍼 개수" 기준 분할이었고, split 경계에 걸친 lot 하나가 train/test 양쪽에 나뉘어 들어가는
**lot 누수**가 있었다(실제로 이전 버전은 `train_lot_range=[1, 46088]`, `test_lot_range=[46088,
47542]`로 lot 46088이 양쪽에 동시에 존재했다). 고유 `lot_num` 값을 먼저 정렬한 뒤, 그 목록을
80/20으로 나누고 각 lot의 모든 웨이퍼를 통째로 한쪽에만 배정하도록 수정했다 — `train_lots`와
`test_lots`가 서로소(disjoint)임을 assert로 확인한다.

### 재검증 방법

- 고유 lot을 정렬 후 앞 80% / 뒤 20%로 분할 (수정 후: train lot1~45283, test lot45284~47542).
- train: 4단계와 동일하게 클래스별 최대 2,000장 상한. test: 클래스별 최대 500장 상한 —
  500장 미만인 클래스는 그 구간에 있는 만큼만 쓰고 업샘플링하지 않지만, 500장을 넘는 클래스
  (none 등)는 여전히 500으로 잘리므로 **이 test set의 클래스 비율이 실제 생산 현장 비율을
  그대로 반영하는 것은 아니다** — Accuracy를 "실전 배포 정확도"로 해석하면 안 되는 이유다.
- 모델: `HybridCNN`(CNN + 8개 파생변수), 나머지 하이퍼파라미터는 4단계와 동일.

### 결과 (버그 수정 후)

| 분할 | Macro F1 | Accuracy |
|---|---|---|
| 무작위 (기존, 4단계) | 0.8436 | - |
| lot 순서 (수정 후) | **0.6031** | 0.5685 |

test 클래스별 실제 표본 수: none 500, Center 500, Edge-Loc 500, Loc 500, Edge-Ring 363,
Scratch 347, Random 114, Near-full 38, **Donut 30**.

클래스별 F1: `none` 0.561, `Center` 0.726, `Donut` 0.588, `Edge-Loc` 0.588, `Edge-Ring` 0.369,
`Loc` 0.598, `Random` 0.771, `Scratch` 0.298, `Near-full` 0.930.

### 해석

완전히 0으로 붕괴하지는 않았다 — 웨이퍼맵의 결함 패턴(모양)은 비교적 강건한 시각적 특징이기
때문으로 보인다. 그러나 Macro F1이 0.84→0.60으로 하락한 것은 무시할 수 없는 수준이며, 특히
Edge-Ring(0.96→0.37)처럼 무작위 분할에서는 쉬웠던 클래스가 이 구간에서 크게 무너지는 등, 어떤
클래스가 취약한지 자체가 분할 방식에 따라 달라진다. Near-full처럼 시각적으로 매우 뚜렷한 패턴은
분할 방식과 무관하게 안정적으로 높은 성능을 유지한다.

## 2x2 조합 + 부트스트랩 신뢰구간 (`src/ablation_with_ci.py`)

### 동기

4~5단계(`derived_features.py`, `augmentation.py`)는 각각 독립된 프로세스 실행이었다. 이 때문에
(1) "증강만, 파생변수는 없이" 조합이 실험에서 누락돼 있었고, (2) CPU 스레드 비결정성으로 같은
조건의 baseline 수치가 스크립트마다 0.4~0.8%p씩 어긋났으며, (3) 개선폭이 통계적으로 유의미한지
확인되지 않았다. 네 조합을 한 프로세스·동일 데이터 분할에서 전부 재학습해 이 세 가지를 동시에
해결했다.

### 방법

- `train_test_split(stratify, random_state=42)`로 고정한 하나의 test set(2,553장)에 대해,
  baseline / derived / augment / both 네 모델을 순서대로 학습·예측.
- test set 예측 결과를 1,000회 부트스트랩 재추출해 각 조건의 macro F1 신뢰구간과, 조건 간
  델타(개선폭)의 신뢰구간을 함께 계산 (paired bootstrap — 같은 리샘플 인덱스로 두 조건을 동시에
  비교해 대응표본 검정에 해당).
- `torch.set_num_threads(1)` 적용 상태로 실행해 재현성 확보 (총 소요 45.6분).

### 결과

| 조합 | Macro F1 | 95% CI |
|---|---|---|
| baseline | 0.7824 | 0.7648 ~ 0.7986 |
| derived | 0.8363 | 0.8206 ~ 0.8504 |
| augment | 0.8428 | 0.8268 ~ 0.8581 |
| both | 0.8932 | 0.8810 ~ 0.9035 |

| 비교 | 평균 델타 | 95% CI | 유의미 |
|---|---|---|---|
| derived − baseline | +0.0540 | +0.0351 ~ +0.0731 | Yes |
| augment − baseline | +0.0605 | +0.0423 ~ +0.0775 | Yes |
| both − baseline | +0.1108 | +0.0938 ~ +0.1283 | Yes |
| both − derived | +0.0569 | +0.0438 ~ +0.0714 | Yes |
| both − augment | +0.0504 | +0.0374 ~ +0.0644 | Yes |

다섯 비교 모두 95% CI가 0을 포함하지 않아 통계적으로 유의미하다. augment 단독(+0.0605)이
derived 단독(+0.0540)보다 근소하게 크며, both가 derived·augment 각각보다도 유의미하게 높아
(+0.0569, +0.0504) 두 방법이 상호보완적임을 시사한다 — 단순 합(0.0540+0.0605=0.1145)에는
살짝 못 미치지만(실제 +0.1108), 오차범위 내에서 거의 가산적인 효과로 볼 수 있다.

### 한계: 이 실험은 test set을 모델 선택에도 사용한다

위 실험은 네 조합을 **같은 test set**에서 비교하고, 그중 가장 높은 조합("both")을 그대로 최종
성능으로 제시했다. 이 방식은 "test에서 가장 잘 나온 조합을 고른 뒤, 그 test 점수를 최종 성능
이라고 보고하는" 구조라서, test set이 사실상 모델 선택에도 관여한 셈이 된다 — 여러 후보 중
최댓값을 취하면 그 최댓값은 미래의 새 데이터에서보다 낙관적으로 나올 수 있다(선택 편향,
winner's curse). 이 문제를 해결하기 위해 아래 `final_evaluation.py`로 train/validation/test
3-way 분리를 도입했다. 이 섹션의 수치는 "네 조합의 상대적 우열"을 확인하는 참고 자료로만
사용하고, **공식 최종 성능은 다음 섹션의 test 수치를 사용한다.**

## Train/Val/Test 3-way 분리 최종 평가 (`src/final_evaluation.py`)

### 동기

바로 위 `ablation_with_ci.py`의 한계(모델 선택과 최종 평가에 같은 test set을 재사용)를 해소하기
위해, 데이터를 train(60%)/validation(20%)/test(20%)로 분리했다. "어느 조합이 최선인가"는
validation에서만 결정하고, 최종 성능은 그 결정에 전혀 관여하지 않은 test set에서 딱 한 번만
측정한다.

### 방법

1. `train_test_split(test_size=0.4, stratify, random_state=42)`로 train(60%)과 rest(40%)를 분리한
   뒤, rest를 다시 절반으로 나눠 validation(20%)/test(20%)를 만든다 (`train=7,657`,
   `val=2,553`, `test=2,553`).
2. baseline/derived/augment/both 네 조합을 train으로 학습하고 **validation**에서 macro F1과
   부트스트랩 신뢰구간을 계산해 최선의 조합을 고른다.
3. 최선 조합(용도 확정 후)만 **train+validation**으로 다시 학습시키고, **test**에서 딱 한 번
   평가해 macro F1과 신뢰구간을 계산한다.
4. 최종 모델의 `state_dict`, 스케일러, test 인덱스를 저장(`results/models/`)해 `grad_cam.py`가
   재학습 없이 동일 모델·동일 test split을 재사용하도록 한다.
5. `torch.set_num_threads(1)` 적용 (총 소요 51.6분: validation 선택 단계 4회 학습 + 최종 재학습
   1회).

### 결과 — 1단계: validation 기준 조합 비교 (모델 선택용)

| 조합 | Macro F1 | 95% CI |
|---|---|---|
| baseline | 0.7990 | 0.7827 ~ 0.8141 |
| derived | 0.8297 | 0.8141 ~ 0.8462 |
| augment | 0.8016 | 0.7865 ~ 0.8166 |
| **both** | **0.8705** | 0.8575 ~ 0.8845 |

validation 기준으로도 "both"(파생변수+증강)가 가장 우수해, `ablation_with_ci.py`의 결론과
일관된다. 다만 이번엔 derived(+0.0307)가 augment(+0.0026)보다 뚜렷하게 크게 나와 — 어떤 20%가
validation으로 빠지는지에 따라 "파생변수 단독 vs 증강 단독" 중 어느 쪽이 더 큰 효과인지의
순위 자체는 흔들릴 수 있음을 보여준다. "둘 다 쓰는 것이 최선"이라는 결론만은 두 번의 독립적인
실행 모두에서 일관되게 유지된다. (주의: 위는 조건별 개별 신뢰구간을 나열한 것이지, 조건 간
델타의 신뢰구간을 직접 계산한 것은 아니다 — 그 방식은 다음 섹션 `ablation_with_ci.py`의 paired
bootstrap delta CI를 참고.)

### 결과 — 2단계: 최종 test 평가 (단 한 번)

validation에서 확정된 "both" 조합을 train+validation(36,880장, 증강 포함)으로 재학습해 test
(2,553장)에서 평가했다.

| 지표 | 값 |
|---|---|
| Macro F1 | **0.8851** (95% CI 0.8676~0.8996) |
| Accuracy | 0.8844 |

클래스별 F1: `none` 0.878, `Center` 0.953, `Donut` 0.937, `Edge-Loc` 0.835, `Edge-Ring` 0.975,
`Loc` 0.807, `Random` 0.924, `Scratch` 0.780, `Near-full` 0.877.

`ablation_with_ci.py`가 보고했던 0.8932(test 재사용)보다 소폭 낮지만 신뢰구간은 크게 겹친다
(0.8676~0.8996 vs 0.8810~0.9035) — 이번 사례에서는 선택 편향의 영향이 크지 않았다는 뜻이지만,
일반적으로는 이 차이가 더 클 수 있으므로 두 값을 구분해 보고하는 습관 자체가 중요하다.

### 한계

신뢰구간은 테스트셋 재표본추출 변동만 반영하며, 재학습 시드 변동이나 train/val/test 분할 자체가
달라졌을 때의 변동은 포함하지 않는다.

**더 중요한 한계: 이 분할은 lot 기준이 아니다.** 위 train(60%)/val(20%)/test(20%)는 클래스별
상한을 둔 subset을 **웨이퍼 단위로 무작위** 분할한 것이라, 같은 lot의 웨이퍼가 세 split에
나뉘어 들어갈 수 있다. `lot_split_validation.py`에서 이미 확인했듯 lot 순서에 따라 불량
비율이 극단적으로 다르므로(초반 20% lot 약 51% vs 나머지 3~10%), 같은 lot이 여러 split에
섞이면 "신규 lot에 대한 일반화"를 과대평가할 위험이 있다. 즉 위 0.8851은 **"같은 분포
안에서 무작위로 나눴을 때의 성능"** 이지 "신규 lot에서의 일반화 성능"이 아니다. 이 문제를
lot 기준으로 다시 해결한 것이 다음 섹션이다.

## lot 기준 Train/Val/Test 최종 평가 (`src/lot_final_evaluation.py`)

### 동기

`lot_split_validation.py`(고유 lot 기준 두 갈래 분할)와 `final_evaluation.py`(모델 선택/최종
평가 분리)는 각각 다른 문제를 해결했지만, 두 문제를 동시에 해결한 적은 없었다 —
`lot_split_validation.py`는 lot 누수는 없지만 사전에 고정한 단일 모델을 train/test 2-way로
평가한 진단 실험이라 validation 기반 모델 선택 절차가 없고, `final_evaluation.py`는 모델
선택과 최종 평가를 분리했지만 lot 기준이 아니다. 이 스크립트는 두 방법론을 합쳐 **lot 단위로
train(60%)/validation(20%)/test(20%)를 나누고, 그 안에서 모델 선택과 최종 평가를 분리**한다.

### 방법

1. 고유 `lot_num`을 정렬해 앞 60%/다음 20%/마지막 20%를 train/val/test lot으로 지정
   (`lot_split_validation.py`와 동일한 lot 기반 분리 방식 — `assert`로 세 집합이 서로소임을
   확인).
2. train은 클래스당 최대 2,000장으로 제한하고 소수 클래스에 증강을 적용. val/test는 증강
   없이 클래스당 최대 500장을 고정 시드로 무작위 추출한다(업샘플링 없음). 따라서 lot은
   분리되지만 실제 생산 클래스 분포를 보존한 평가는 아니다.
3. 4개 조합(baseline/derived/augment/both)을 train으로 학습해 **validation**에서 비교, 최선의
   조합을 결정 (test는 이 단계에서 보지 않음).
4. 최선 조합만 train+validation으로 재학습해 **test에서 딱 한 번** 평가.
5. 신뢰구간은 웨이퍼가 아니라 lot을 재표본추출하는 cluster bootstrap으로 계산.
6. 최종 모델·스케일러·test subset을 저장해 `grad_cam.py`가 이 모델을 재사용하도록 갱신
   (이전에 저장했던 웨이퍼 단위 무작위 분할 모델을 대체).

### 결과

lot 범위: train lot1~42995(6,457개 lot, subset 11,922장) / validation lot42996~45283(2,152개
lot, subset 2,456장) / test lot45284~47542(2,153개 lot, subset 2,892장).

**1단계: validation(신규 lot) 기준 조합 비교**

| 조합 | Macro F1 |
|---|---|
| baseline | 0.7058 |
| derived | 0.7149 |
| augment | 0.7040 |
| **both** | **0.7847** |

무작위 분할(7단계 validation)과 마찬가지로 "both"가 가장 우수했다. 다만 절대 수치는 전반적으로
더 낮다(baseline 0.7058 vs 무작위 분할의 0.7990) — 신규 lot에서는 문제 자체가 더 어렵다는
뜻이다.

**2단계: 최종 test(더 새로운 lot) 평가 — 딱 한 번**

validation에서 확정된 "both" 조합을 train+validation(47,516장, 증강 포함)으로 재학습해 test
(2,892장)에서 평가했다.

| 지표 | 값 |
|---|---|
| Macro F1 | **0.7207** |
| Accuracy | 0.7095 (class-capped subset 참고값) |

클래스별 F1: `none` 0.7277, `Center` 0.8300, `Donut` 0.5417, `Edge-Loc` 0.6940, `Edge-Ring`
0.6026, `Loc` 0.7120, `Random` 0.8509, `Scratch` 0.6009, `Near-full` 0.9268.

### 해석

0.7207을 클래스 상한이 적용된 신규 lot holdout의 대표 수치로 사용한다. 신뢰구간은 같은 lot
안의 웨이퍼를 독립 표본으로 간주하지 않도록 lot 단위 cluster bootstrap으로 계산한다. 다만
재학습 시드 변동과 실제 생산 클래스 분포는 반영하지 않는다. 6단계(`lot_split_validation.py`)의
2-way 진단 결과(0.603), 7단계(`final_evaluation.py`)의 IID 성능(0.8851)과 함께 정리하면:

| 스크립트 | 분할 기준 | 모델 선택/평가 분리 | Macro F1 | 답하는 질문 |
|---|---|---|---|---|
| `final_evaluation.py` | 웨이퍼 무작위 | O | 0.8851 | 같은 분포에서 패턴을 얼마나 잘 배우는가 |
| `lot_split_validation.py` | lot 기준 | X (2-way) | 0.603 | 사전 고정 모델의 신규 lot 진단 결과 |
| `lot_final_evaluation.py` | lot 기준 | O (3-way) | **0.7207** | validation 선택 후 독립 신규 lot 성능 |

클래스별로는 Near-full(0.927)·Random(0.851)·Center(0.830)처럼 패턴이 뚜렷한 클래스는 여전히
높았지만, Edge-Ring(0.603)·Donut(0.542)·Scratch(0.601)처럼 형태가 미세하거나 이 test lot
구간에서 표본 비율이 달라진 클래스는 낮았다. 포트폴리오에서는 0.8851을 동일 분포 성능,
0.7207을 class-capped 신규 lot holdout 성능으로 함께 제시한다.

이 수치가 6단계(`lot_split_validation.py`)의 0.603, 7단계(`final_evaluation.py`)의 0.8851과
어떻게 다른 질문에 답하는지 정리하면:

| 스크립트 | 분할 기준 | 모델 선택/평가 분리 | 답하는 질문 |
|---|---|---|---|
| `final_evaluation.py` | 웨이퍼 무작위 | O | 같은 분포에서 패턴을 얼마나 잘 배우는가 |
| `lot_split_validation.py` | lot 기준 | X (2-way) | 사전 고정 모델의 신규 lot 진단 결과 |
| `lot_final_evaluation.py` | lot 기준 | O (3-way) | validation 선택 후 독립 신규 lot 성능 |

## Grad-CAM 설명력 분석 (`src/grad_cam.py`)

### 동기

지금까지는 F1 점수로만 모델을 평가했는데, 점수가 높다고
모델이 항상 올바른 근거로 판단한다는 보장은 없다. Grad-CAM으로 CNN이 예측 시 이미지의 어느
공간 영역에 반응했는지 직접 확인했다.

### 한계: 이 모델의 두 번째 경로는 Grad-CAM으로 보이지 않는다

최종 모델(`HybridCNN`)은 CNN 이미지 경로의 flatten 출력에 파생변수 8개를 concat한 뒤
분류기에 넣는 구조다. 여기서 사용하는 Grad-CAM은 CNN 경로의 마지막 conv 레이어에만 hook을
걸기 때문에, **파생변수 경로가 최종 판단에 기여한 정도는 이 기법으로 전혀 설명되지 않는다.**
아래 히트맵은 "이미지 정보만 놓고 봤을 때 모델이 어디에 반응했는가"로 읽어야 하며, 모델
전체의 판단 근거를 나타내는 것은 아니다.

### 방법

- final_evaluation.py가 train/val/test 3-way 분리로 선택하고 test에서 한 번만 평가한 최종
  모델(`results/models/final_model.pt`)과 그때 쓴 동일한 test split을 그대로 재사용한다 —
  이 스크립트를 위해 별도로 재학습하지 않는다(재학습할 때마다 예시가 바뀌는 것을 방지).
- `HybridCNN.features[7]`(마지막 Conv2d+ReLU, AdaptiveAvgPool2d 이전 — 16×16 공간 해상도가
  남아있는 마지막 지점)에 forward/backward hook을 걸어 활성값과 그레디언트를 취득.
- 표준 Grad-CAM 공식: 채널별 그레디언트를 공간 평균해 가중치로 사용, 가중합 후 ReLU, 입력
  해상도(64×64)로 bilinear 업샘플.
- 9개 클래스 각각에서 정확히 분류된 사례를 **무작위로** 1건씩 뽑아 시각화
  (`21_grad_cam_by_class.png`, 이전 버전은 "가장 먼저 맞은 사례"를 고정으로 뽑아 우연히
  쉬운 예시로 편향될 수 있었다), Scratch는 추가로 4건을 무작위로 뽑아 상세 비교
  (`22_grad_cam_scratch_detail.png`).

### 결과

(lot 기준 최종 모델로 갱신된 결과) 클래스별 대표 사례(무작위 추출, test index 표기):
`none`#40(p=0.77), `Center`#891(p=1.00), `Donut`#1018(p=0.53), `Edge-Loc`#1255(p=0.91),
`Edge-Ring`#1703(p=0.52), `Loc`#2324(p=0.86), `Random`#2402(p=1.00), `Scratch`#2740(p=0.54),
`Near-full`#2861(p=1.00) — 전부 정답. Random/Near-full은 넓은 영역에 반응, Loc은 한 곳에
뭉친 국소 반응, Edge-Loc/Edge-Ring은 테두리 근처에 반응 — 각 패턴의 정의와 대체로 부합한다.
이번 Scratch 예시(#2740)는 선 전체가 아니라 한 지점의 국소적인 blob에만 강하게 반응했다.
`none`(정상) 사례(#40)에서도 국소적으로 강한 반응이 관찰됐으나 원인은 이번 분석 범위에서
확정하지 못했다.

Scratch 4건 상세(무작위 추출): 정답(#2539, #2688)은 대각선 방향으로 늘어선 국소 반응이
관찰돼 선의 일부 구간을 부분적으로 따라가는 모습이었지만 선 전체를 깔끔하게 추적하지는
못했고, 정답(#2844)은 선이 아니라 웨이퍼 여러 지점에 흩어진 반응을 보였다. 오답(#2762,
실제 예측은 none)은 선이 아닌 국소 blob 하나에 반응했다. 4건 중 선 전체를 뚜렷하게 추적한
사례는 없었다 — lot 기준으로 재학습한 모델에서도 동일한 패턴(부분적/국소적 단서 의존)이
재현된다.

### 해석

Scratch F1이 지속적으로 낮은 원인이 단순히 학습 표본 부족만이 아니라, **정답을 맞힌 경우에도
모델이 매번 일관된 근거(실제 결함 선 전체)로 판단하지 않고 있다**는 것을 시각적으로 확인했다.
이는 정확도/F1 지표만으로는 드러나지 않는 신뢰성 문제이며, 정량 지표와 설명력 분석을 함께 봐야
모델의 실제 견고성을 판단할 수 있다는 근거가 된다.

## 재현 방법

```bash
bash run_all.sh
```
내부적으로 `data/raw/`에 zip을 다운로드 → 압축 해제 → `pip install` →
`src/eda.py` → `src/train_cnn.py` → `src/derived_features.py` → `src/augmentation.py` →
`src/lot_drift.py` → `src/lot_split_validation.py` → `src/ablation_with_ci.py` →
`src/final_evaluation.py` → `src/lot_final_evaluation.py` → `src/grad_cam.py` 순서로 실행됩니다.

`ablation_with_ci.py`와 `final_evaluation.py`는 둘 다 4개 조합(baseline/derived/augment/both)을
재학습합니다 — 전자는 참고용 탐색 실험(모델 선택과 최종 평가에 같은 test set 재사용, 위 한계
참고), 후자가 train/val/test 3-way 분리로 이 문제를 해결했지만 여전히 웨이퍼 단위 무작위
분할입니다("같은 분포에서의 성능" 수치). `lot_final_evaluation.py`가 lot 기준 3-way 분리로
class-capped 신규 lot holdout 대표 수치를 낸다 — **`grad_cam.py`는 이 스크립트가 저장한 모델을
재사용**하므로(`final_evaluation.py`가 먼저 저장한 모델을 나중에 덮어씀) 재학습하지 않습니다.

## 프로젝트 구조

```
wm811k-defect-classification/
├── data/
│   └── raw/                     # 원본 zip + 압축 해제 (run_all.sh가 자동 다운로드)
├── assets/fonts/                  # 한글 렌더링용 NanumGothic (plot_style.py가 사용)
├── src/
│   ├── plot_style.py              # matplotlib 한글 폰트 공용 설정
│   ├── data.py                    # pkl 로드 + 라벨 필터링
│   ├── eda.py                     # 클래스 분포, 크기 분포, 샘플 시각화
│   ├── train_cnn.py               # 전처리 + CNN 학습 + 평가
│   ├── derived_features.py        # 파생변수 설계 + 유무 비교 실험
│   ├── augmentation.py            # 회전/반전 증강 실험
│   ├── lot_drift.py               # lot 순서 기반 drift 분석
│   ├── lot_split_validation.py    # lot 순서 분할 재검증 (고유 lot 기준, 2-way, 탐색용)
│   ├── ablation_with_ci.py        # 파생변수x증강 2x2 조합 + 부트스트랩 신뢰구간 (탐색용)
│   ├── final_evaluation.py        # train/val/test 3-way (웨이퍼 무작위) — 동일 분포 성능
│   ├── lot_final_evaluation.py    # train/val/test 3-way (lot 기준) — 신규 lot holdout 대표 성능
│   └── grad_cam.py                # Grad-CAM 설명력 분석 (lot_final_evaluation의 모델 재사용)
├── results/
│   ├── figures/                    # 01~24 시각화 결과
│   ├── models/                     # 최종 평가 스크립트가 저장한 최종 모델·스케일러
│   └── *.json, *.csv               # 수치 결과
├── requirements.txt
└── run_all.sh
```
