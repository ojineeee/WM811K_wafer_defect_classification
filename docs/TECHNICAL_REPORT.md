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

[SECOM 프로젝트](https://github.com/ojineeee/SECOM_defect_prediction)에서 무작위 분할이 시간
드리프트를 감추고 있었다는 걸 확인한 뒤, 이 프로젝트에도 같은 위험이 있는지 점검이 빠져 있었다는
걸 뒤늦게 발견했다. WM-811K에는 타임스탬프가 없지만 `lotName`이 `lot{N}` 형식의 순번이라 생산
순서의 proxy로 사용했다.

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

초반 2개 분위(전체의 20%) 평균 0.5071 vs 나머지 8개 분위 평균 0.0576 — **약 8.8배 차이**로,
SECOM에서 관찰된 드리프트(8.56%→4.72%, 약 1.8배)보다 훨씬 극단적이다. lot 순서 자체가 균등
간격의 시간이 아니라는 점(초반 lot일수록 결함 비율이 높다는 건 라벨링 정책 혹은 실제 초기 공정
불안정성 둘 다 가능한 원인이며, 메타데이터만으로는 구분 불가)에 유의해야 한다.

### 재검증 방법

- `df.sort_values('lot_num')` 후 행 순서 기준 앞 80% / 뒤 20%로 분할 (SECOM의 시간순 분할과
  동일한 논리, 개별 lot을 쪼개지는 않음 — 한 lot 내 wafer는 모두 같은 쪽에 포함).
- train: 4단계와 동일하게 클래스별 최대 2,000장 상한. test: 클래스별 최대 500장 상한이지만
  **미래(뒤 20%) 구간에 자연적으로 존재하는 만큼만** 사용 — 업샘플링하지 않음.
- 모델: `HybridCNN`(CNN + 8개 파생변수), 나머지 하이퍼파라미터는 4단계와 동일.

### 결과

| 분할 | Macro F1 | Accuracy |
|---|---|---|
| 무작위 (기존, 4단계) | 0.8436 | - |
| lot 순서 | **0.6110** | 0.6434 |

test 클래스별 실제 표본 수(자연 분포): none 500, Edge-Loc 500, Loc 356, Scratch 239,
Edge-Ring 229, Center 92, Random 37, Near-full 16, **Donut 5**. Donut처럼 표본이 5개뿐인
클래스는 F1 해석에 주의가 필요하다(단 1~2건의 오분류로도 크게 흔들림).

클래스별 F1: `none` 0.784, `Center` 0.719, `Donut` 0.286, `Edge-Loc` 0.674, `Edge-Ring` 0.642,
`Loc` 0.596, `Random` 0.581, `Scratch` 0.276, `Near-full` 0.941.

### 해석

SECOM처럼 완전히 0으로 붕괴하지는 않았다 — 웨이퍼맵의 결함 패턴(모양)은 센서 값보다 시간에
더 강건한 시각적 특징이기 때문으로 보인다. 그러나 Macro F1이 0.84→0.61로 하락한 것은 무시할
수 없는 수준이며, 특히 원래도 어려웠던 Scratch(0.37→0.28 수준)·Donut처럼 형태가 미세하거나
표본이 적은 클래스에서 하락 폭이 크다. Near-full처럼 시각적으로 매우 뚜렷한 패턴은 분할 방식과
무관하게 안정적으로 높은 성능을 유지한다.

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

## Grad-CAM 설명력 분석 (`src/grad_cam.py`)

### 동기

SECOM의 SHAP과 대칭되는 분석. 지금까지는 F1 점수로만 모델을 평가했는데, 점수가 높다고
모델이 항상 올바른 근거로 판단한다는 보장은 없다. Grad-CAM으로 CNN이 예측 시 이미지의 어느
공간 영역에 반응했는지 직접 확인했다.

### 방법

- 최고 성능 조합("both" = 파생변수 + 증강)을 동일 하이퍼파라미터로 재학습.
- `HybridCNN.features[7]`(마지막 Conv2d+ReLU, AdaptiveAvgPool2d 이전 — 16×16 공간 해상도가
  남아있는 마지막 지점)에 forward/backward hook을 걸어 활성값과 그레디언트를 취득.
- 표준 Grad-CAM 공식: 채널별 그레디언트를 공간 평균해 가중치로 사용, 가중합 후 ReLU, 입력
  해상도(64×64)로 bilinear 업샘플.
- 9개 클래스 각각에서 정확히 분류된 사례 1건씩 시각화(`21_grad_cam_by_class.png`), Scratch는
  추가로 4건을 뽑아 상세 비교(`22_grad_cam_scratch_detail.png`).

### 결과

클래스별 대표 사례: Scratch는 실제 결함 선을 따라 히트맵이 정확히 일치, Random/Near-full은
넓은 영역에 반응, Center/Donut/Edge-Ring/Loc은 각 패턴의 정의와 부합하는 위치에 반응했다.
`none`(정상) 사례 일부에서도 국소적으로 강한 반응이 관찰됐으나 원인은 이번 분석 범위에서
확정하지 못했다.

Scratch 4건 상세: 정답(#15)은 선을 정확히 추적했지만, 정답(#29, #37)은 각각 웨이퍼 가장자리와
분산된 지점에 반응했고, 오답(#9, 실제 예측은 Edge-Loc)은 선이 아닌 뭉친 영역에 반응했다.
4건 중 선을 명확히 추적한 것은 1건뿐이다.

### 해석

Scratch F1이 지속적으로 낮은 원인이 단순히 학습 표본 부족만이 아니라, **정답을 맞힌 경우에도
모델이 매번 일관된 근거(실제 결함 선)로 판단하지 않고 있다**는 것을 시각적으로 확인했다. 이는
정확도/F1 지표만으로는 드러나지 않는 신뢰성 문제이며, 정량 지표와 설명력 분석을 함께 봐야
모델의 실제 견고성을 판단할 수 있다는 근거가 된다.

## 재현 방법

```bash
bash run_all.sh
```
내부적으로 `data/raw/`에 zip을 다운로드 → 압축 해제 → `pip install` →
`src/eda.py` → `src/train_cnn.py` → `src/derived_features.py` → `src/augmentation.py` →
`src/lot_drift.py` → `src/lot_split_validation.py` → `src/ablation_with_ci.py` → `src/grad_cam.py`
순서로 실행됩니다.

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
│   ├── lot_split_validation.py    # lot 순서 분할 재검증
│   ├── ablation_with_ci.py        # 파생변수x증강 2x2 조합 + 부트스트랩 신뢰구간
│   └── grad_cam.py                # Grad-CAM 설명력 분석
├── results/
│   ├── figures/                    # 01~22 시각화 결과
│   └── *.json, *.csv               # 수치 결과
├── requirements.txt
└── run_all.sh
```
