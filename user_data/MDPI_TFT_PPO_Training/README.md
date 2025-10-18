# MDPI TFT + PPO 통합 학습 시스템

이 시스템은 **MDPI 스타일의 TFT 사전학습 + PPO 파인튜닝**을 통합한 완전한 학습 파이프라인입니다:

## 🎯 주요 특징
- **MDPI 정규화**: z/log/logit 정규화 (train에만 fit, per asset)
- **MultiTaskTFT**: 24h 수익률 예측 + 방향성/변동성 보조 태스크
- **PPO 파인튜닝**: TFT 인코더를 활용한 강화학습 정책 최적화
- **Optuna 하이퍼파라미터 튜닝**: 자동 최적화
- **재현 가능한 체크포인트**: 스케일러 + 설정 포함

## 📁 파일 구조
```
MDPI_TFT_PPO_Training/
├── configs/
│   ├── mdpi_tft.yml          # 기존 MDPI TFT 설정
│   └── mdpi_tft_ppo.yml      # 통합 TFT+PPO 설정
├── scripts/
│   ├── train_mdpi_tft.py     # 기존 MDPI TFT 학습
│   ├── train_mdpi_pipeline.py # 통합 TFT+PPO 학습
│   ├── run_training.py       # 실행 스크립트
│   └── mdpi_normalization.py # MDPI 정규화 모듈
└── README.md
```

## 🚀 사용법

### 1) 데이터 준비
`user_data/datasets/ohlcv.csv` 파일을 준비하세요:
```csv
date,open,high,low,close,volume,asset
2023-01-01 00:00:00,1000,1050,990,1020,1000000,BTC/USDT
2023-01-01 01:00:00,1020,1030,1010,1025,950000,BTC/USDT
...
```

### 2) 설정 수정
`configs/mdpi_tft_ppo.yml`에서 다음을 수정하세요:
- `data.ohlcv_path`: 데이터 파일 경로
- `data.asset`: 거래할 자산
- `split.train_val_cut`: TFT 학습/검증 분할 날짜
- `data.finetune_days`: PPO 파인튜닝용 최근 데이터 일수

### 3) 학습 실행
```bash
# 통합 학습 (TFT + PPO)
python user_data/MDPI_TFT_PPO_Training/scripts/run_training.py

# 또는 직접 실행
python user_data/MDPI_TFT_PPO_Training/scripts/train_mdpi_pipeline.py

# 설정 파일 지정
python user_data/MDPI_TFT_PPO_Training/scripts/run_training.py \
  --config user_data/MDPI_TFT_PPO_Training/configs/mdpi_tft_ppo.yml
```

### 4) 출력 파일
학습 완료 후 다음 파일들이 생성됩니다:
- `user_data/models/tft_encoder.pt` - TFT 인코더 (PPO용)
- `user_data/models/ppo_policy.zip` - PPO 정책
- `user_data/models/best/mdpi_tft.pt` - MDPI TFT 모델
- `user_data/models/best/mdpi_scaler.pkl` - 정규화 스케일러
- `user_data/models/best/mdpi_config.json` - 설정 파일

## 🔧 고급 설정

### TFT 모델 설정
```yaml
tft:
  enc_len: 64              # 인코더 길이 (시간 윈도우)
  hidden_size: 160          # 히든 사이즈
  attention_heads: 4       # 어텐션 헤드 수
  dropout: 0.2             # 드롭아웃 비율
  horizons: [24, 48, 96]   # 멀티-호라이즌 타깃
  max_epochs: 50           # 최대 에폭 수
  early_stopping:
    patience: 8            # 조기종료 인내심
```

### PPO 모델 설정
```yaml
ppo:
  timesteps: 100000        # 총 학습 스텝 수
  learning_rate: 3e-4      # 학습률
  n_steps: 2048           # 업데이트당 스텝 수
  batch_size: 64          # 배치 사이즈
  gamma: 0.99             # 할인 인수
  clip_range: 0.2         # 클리핑 범위
```

## 📊 성능 목표
- **Sharpe Ratio**: > 1.5 (목표: 2.0+)
- **Win Rate**: 55-70%
- **Max Drawdown**: ≤ 15%
- **CAGR**: 10-25%

## 🔍 모니터링
학습 중 다음 메트릭들이 출력됩니다:
- TFT: train_loss, val_loss, AUC, IC@20, VolCorr
- PPO: trained_steps, entropy
- 최종: Sharpe, Sortino, Calmar, MDD, WinRate

## 🛠️ 문제 해결
1. **메모리 부족**: `batch_size`를 줄이세요
2. **학습 속도**: `num_workers`를 늘리세요 (Windows에서는 0 권장)
3. **수렴 문제**: `learning_rate`를 조정하세요
4. **과적합**: `dropout`을 늘리거나 `early_stopping.patience`를 줄이세요
