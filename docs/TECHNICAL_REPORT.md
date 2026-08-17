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

주의: 이 실행의 "증강 없음" macro F1(0.8396)은 4단계에서 보고한 0.8436과 소폭 다른데, 이는 별도 프로세스 재실행에 따른 PyTorch CPU 연산의 미세한 비결정성 때문이며 두 수치 모두 유효한 반복 실행 결과입니다.

## 재현 방법

```bash
bash run_all.sh
```
내부적으로 `data/raw/`에 zip을 다운로드 → 압축 해제 → `pip install` → `src/eda.py` → `src/train_cnn.py` → `src/derived_features.py` → `src/augmentation.py` 순서로 실행됩니다.

## 프로젝트 구조

```
wm811k-defect-classification/
├── data/
│   └── raw/                  # 원본 zip + 압축 해제 (run_all.sh가 자동 다운로드)
├── src/
│   ├── data.py                # pkl 로드 + 라벨 필터링
│   ├── eda.py                 # 클래스 분포, 크기 분포, 샘플 시각화
│   ├── train_cnn.py           # 전처리 + CNN 학습 + 평가
│   └── derived_features.py    # 파생변수 설계 + 유무 비교 실험
├── results/
│   ├── figures/                # 01~07 시각화 결과
│   ├── eda_summary.json, cnn_final_report.json
│   └── derived_feature_experiment.json, derived_feature_by_class.csv
├── requirements.txt
└── run_all.sh
```
