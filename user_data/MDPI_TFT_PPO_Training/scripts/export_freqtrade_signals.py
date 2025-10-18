#!/usr/bin/env python3
"""
MDPI TFT 모델로 예측→z-score→CSV 생성 스크립트
학습된 모델을 사용해서 freqtrade 전략에서 사용할 신호를 생성합니다.
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import pickle
import json
from datetime import datetime

# 현재 스크립트 디렉토리를 Python path에 추가
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
sys.path.append(os.path.join(script_dir, '..', '..'))

from TFT_PPO_modules.feature_pipeline import FeaturePipeline
from TFT_PPO_modules.multi_task_tft import MultiTaskTFT, create_multi_horizon_targets
from pytorch_forecasting.data import TimeSeriesDataSet
from torch.utils.data import DataLoader
from mdpi_normalization import MDPIStandardizer


def load_model_and_scaler(checkpoint_path, scaler_path, config_path):
    """학습된 모델과 스케일러 로드"""
    print(f"Loading model from {checkpoint_path}")
    
    # 설정 로드
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 스케일러 로드
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # 모델 로드 (간단한 방법)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 더미 데이터셋으로 모델 초기화
    dummy_df = pd.DataFrame({
        'time_idx': [0, 1, 2],
        'group_id': ['ASSET'] * 3,
        'close': [100, 101, 102],
        'return_24h': [0.01, 0.02, 0.03]
    })
    
    # 피처 컬럼들 추가
    fp = FeaturePipeline()
    dummy_df = fp.add_features(dummy_df)
    dummy_df = create_multi_horizon_targets(dummy_df, horizons=[24, 48, 96])
    
    feature_cols = [c for c in dummy_df.columns if c not in [
        "time_idx", "group_id", "close", "return_24h", "return_48h", "return_96h", 
        "direction", "volatility", "log_return"
    ]]
    
    dummy_ds = TimeSeriesDataSet(
        dummy_df,
        time_idx="time_idx",
        target="return_24h",
        group_ids=["group_id"],
        min_encoder_length=config["tft"]["enc_len"],
        max_encoder_length=config["tft"]["enc_len"],
        min_prediction_length=1,
        max_prediction_length=1,
        static_categoricals=["group_id"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=feature_cols + ["return_24h"],
        add_relative_time_idx=True,
        add_target_scales=False,
        add_encoder_length=True,
    )
    
    # 모델 초기화
    tft = MultiTaskTFT(
        dataset=dummy_ds,
        hidden_size=config["tft"]["hidden_size"],
        attention_head_size=config["tft"]["attention_heads"],
        dropout=config["tft"]["dropout"],
        horizons=[24, 48, 96],
        loss_weights=config["tft"]["loss_weights"],
        device=device
    )
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    tft.load_state_dict(checkpoint)
    tft.eval()
    
    return tft, scaler, config, fp, feature_cols


def predict_returns(df, tft, scaler, fp, feature_cols, window_size=96):
    """데이터프레임에 대해 수익률 예측 수행"""
    print(f"Predicting returns for {len(df)} samples...")
    
    # 피처 엔지니어링
    df = fp.add_features(df)
    df = create_multi_horizon_targets(df, horizons=[24, 48, 96])
    
    # 정규화 적용 (피처만)
    df[feature_cols] = scaler.transform(df, feature_cols)[feature_cols]
    
    # NaN 처리
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # TFT용 인덱싱
    df["time_idx"] = np.arange(len(df))
    df["group_id"] = "ASSET"
    
    predictions = []
    
    # 슬라이딩 윈도우로 예측
    for i in range(window_size, len(df)):
        # 윈도우 데이터 추출
        window_df = df.iloc[i-window_size:i+1].copy()
        
        # TimeSeriesDataSet 생성
        ds = TimeSeriesDataSet(
            window_df,
            time_idx="time_idx",
            target="return_24h",
            group_ids=["group_id"],
            min_encoder_length=window_size,
            max_encoder_length=window_size,
            min_prediction_length=1,
            max_prediction_length=1,
            static_categoricals=["group_id"],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=feature_cols + ["return_24h"],
            add_relative_time_idx=True,
            add_target_scales=False,
            add_encoder_length=True,
        )
        
        # 배치 생성
        dl = ds.to_dataloader(train=False, batch_size=1, num_workers=0)
        
        # 예측
        with torch.no_grad():
            for x, y in dl:
                x = {k: v.to(tft.device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
                pred = tft(x)
                
                # 24h 수익률 예측값 추출
                if isinstance(pred, dict) and "returns" in pred:
                    pred_24h = pred["returns"]["horizon_24"].item()
                else:
                    pred_24h = float(pred.item()) if torch.is_tensor(pred) else float(pred)
                
                predictions.append({
                    'date': df.iloc[i]['date'],
                    'pred_ret_24h': pred_24h
                })
                break
    
    return pd.DataFrame(predictions)


def compute_z_scores(predictions_df, window=96):
    """예측값에 대해 z-score 계산"""
    print(f"Computing z-scores with window={window}...")
    
    z_scores = []
    long_signals = []
    short_signals = []
    
    for i in range(len(predictions_df)):
        if i < window:
            z_score = 0.0
            long_sig = False
            short_sig = False
        else:
            # 롤링 윈도우에서 z-score 계산
            window_preds = predictions_df['pred_ret_24h'].iloc[i-window:i]
            mean_pred = window_preds.mean()
            std_pred = window_preds.std()
            
            if std_pred > 0:
                z_score = (predictions_df['pred_ret_24h'].iloc[i] - mean_pred) / std_pred
            else:
                z_score = 0.0
            
            # 신호 생성
            long_sig = z_score > 0.8   # 상승 신호
            short_sig = z_score < -0.8  # 하락 신호
        
        z_scores.append(z_score)
        long_signals.append(long_sig)
        short_signals.append(short_sig)
    
    predictions_df['z_24h'] = z_scores
    predictions_df['long_sig'] = long_signals
    predictions_df['short_sig'] = short_signals
    
    return predictions_df


def main():
    parser = argparse.ArgumentParser(description="MDPI TFT 예측→z-score CSV 생성")
    parser.add_argument("--data", required=True, help="OHLCV 데이터 파일 경로")
    parser.add_argument("--checkpoint", default="user_data/models/best/mdpi_tft.pt", help="모델 체크포인트 경로")
    parser.add_argument("--scaler", default="user_data/models/best/mdpi_scaler.pkl", help="스케일러 파일 경로")
    parser.add_argument("--config", default="user_data/models/best/mdpi_config.json", help="설정 파일 경로")
    parser.add_argument("--window", type=int, default=96, help="z-score 계산 윈도우 크기")
    parser.add_argument("--long_th", type=float, default=0.8, help="롱 신호 임계값")
    parser.add_argument("--short_th", type=float, default=-0.8, help="숏 신호 임계값")
    parser.add_argument("--out", default="user_data/signals/mdpi_tft_signals.csv", help="출력 CSV 파일 경로")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MDPI TFT 예측→z-score CSV 생성")
    print("=" * 60)
    
    try:
        # 모델 로드
        tft, scaler, config, fp, feature_cols = load_model_and_scaler(
            args.checkpoint, args.scaler, args.config
        )
        
        # 데이터 로드
        print(f"Loading data from {args.data}")
        if args.data.endswith('.feather'):
            df = pd.read_feather(args.data)
        elif args.data.endswith('.csv'):
            df = pd.read_csv(args.data)
        else:
            raise ValueError(f"Unsupported file format: {args.data}")
        
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        
        # 예측 수행
        predictions_df = predict_returns(df, tft, scaler, fp, feature_cols, args.window)
        
        # z-score 계산
        predictions_df = compute_z_scores(predictions_df, args.window)
        
        # freqtrade용 컬럼 추가
        predictions_df['asset'] = 'BTC/USDT'  # 설정에서 가져올 수 있음
        
        # 출력 디렉토리 생성
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        
        # CSV 저장
        predictions_df.to_csv(args.out, index=False)
        
        print("=" * 60)
        print("생성 완료!")
        print("=" * 60)
        print(f"출력 파일: {args.out}")
        print(f"총 샘플 수: {len(predictions_df)}")
        print(f"z-score 통계:")
        print(f"  평균: {predictions_df['z_24h'].mean():.3f}")
        print(f"  표준편차: {predictions_df['z_24h'].std():.3f}")
        print(f"  최대: {predictions_df['z_24h'].max():.3f}")
        print(f"  최소: {predictions_df['z_24h'].min():.3f}")
        print(f"롱 신호 수: {predictions_df['long_sig'].sum()}")
        print(f"숏 신호 수: {predictions_df['short_sig'].sum()}")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
