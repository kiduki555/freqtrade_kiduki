#!/usr/bin/env python3
"""
MDPI TFT + PPO 학습 실행 스크립트
사용법: python train_mdpi_pipeline.py [--config config_path]
"""
import argparse
import sys
import os

# 현재 스크립트 디렉토리를 Python path에 추가
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
sys.path.append(os.path.join(script_dir, '..', '..'))

from train_mdpi_pipeline import train_mdpi_tft_ppo


def main():
    parser = argparse.ArgumentParser(description="MDPI TFT + PPO 통합 학습")
    parser.add_argument(
        "--config", 
        default="user_data/MDPI_TFT_PPO_Training/configs/mdpi_tft_ppo.yml",
        help="설정 파일 경로"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MDPI TFT + PPO 통합 학습 시작")
    print("=" * 60)
    print(f"설정 파일: {args.config}")
    
    try:
        result = train_mdpi_tft_ppo(args.config)
        
        print("=" * 60)
        print("학습 완료!")
        print("=" * 60)
        print(f"최종 Sharpe Ratio: {result['metrics']['sharpe']:.4f}")
        print(f"최종 MDD: {result['metrics']['mdd']:.4f}")
        print(f"최종 Win Rate: {result['metrics']['win_rate']:.2%}")
        print("\n저장된 모델:")
        print("- user_data/models/tft_encoder.pt (TFT 인코더)")
        print("- user_data/models/ppo_policy.zip (PPO 정책)")
        print("- user_data/models/best/mdpi_tft.pt (MDPI TFT)")
        print("- user_data/models/best/mdpi_scaler.pkl (정규화 스케일러)")
        print("- user_data/models/best/mdpi_config.json (설정)")
        
    except Exception as e:
        print(f"학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
