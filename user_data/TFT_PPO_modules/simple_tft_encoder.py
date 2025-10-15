import torch
import torch.nn as nn
import numpy as np
import pandas as pd


class SimpleTFTEncoder(nn.Module):
    """
    훈련 시 사용한 TFT Encoder와 동일한 구조의 간단한 버전
    전체 TFT 모델 대신 encoder 부분만 구현
    """
    
    def __init__(self, input_dim=20, hidden_dim=64, output_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 입력 프로젝션
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # GRU 레이어 (시계열 인코딩)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # 출력 헤드
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        returns: (batch_size, output_dim)
        """
        # 입력 프로젝션
        x = self.input_projection(x)  # (B, T, hidden_dim)
        
        # GRU 인코딩
        gru_out, _ = self.gru(x)  # (B, T, hidden_dim)
        
        # 마지막 타임스텝 사용
        last_hidden = gru_out[:, -1, :]  # (B, hidden_dim)
        
        # 출력 헤드
        output = self.output_head(last_hidden)  # (B, output_dim)
        
        return output


def load_simple_tft_encoder(model_path: str, device: str = "cpu"):
    """
    간단한 TFT Encoder 로드
    """
    try:
        # 모델 생성 (훈련 시와 동일한 파라미터)
        encoder = SimpleTFTEncoder(
            input_dim=20,  # feature_pipeline에서 생성하는 피처 수와 동일
            hidden_dim=64,
            output_dim=64
        )
        
        # state_dict 로드
        checkpoint = torch.load(model_path, map_location=device)
        
        # state_dict 키 매핑 (필요시)
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        # 키 이름이 다를 수 있으므로 strict=False로 로드
        encoder.load_state_dict(state_dict, strict=False)
        encoder.to(device)
        encoder.eval()
        
        return encoder
        
    except Exception as e:
        print(f"SimpleTFTEncoder 로드 실패: {e}")
        return None


def create_fallback_features(df_window: pd.DataFrame) -> np.ndarray:
    """
    TFT 실패 시 사용할 간이 피처 생성
    제로 벡터 대신 의미있는 피처 제공
    """
    if len(df_window) < 2:
        return np.zeros(64, dtype=np.float32)
    
    # 기본 통계 피처
    features = []
    
    # 가격 관련 피처
    if 'close' in df_window.columns:
        close_prices = df_window['close'].values
        returns = np.diff(np.log(close_prices + 1e-8))
        
        # 최근 수익률들
        features.extend([
            np.mean(returns[-5:]) if len(returns) >= 5 else 0,  # 5기간 평균 수익률
            np.std(returns[-10:]) if len(returns) >= 10 else 0,  # 10기간 변동성
            returns[-1] if len(returns) > 0 else 0,  # 최근 수익률
        ])
    else:
        features.extend([0, 0, 0])
    
    # 볼륨 관련 피처
    if 'volume' in df_window.columns:
        volumes = df_window['volume'].values
        features.extend([
            np.mean(volumes[-5:]) / (np.mean(volumes) + 1e-8),  # 최근 볼륨 비율
            np.std(volumes[-10:]) / (np.mean(volumes) + 1e-8),  # 볼륨 변동성
        ])
    else:
        features.extend([0, 0])
    
    # 기술적 지표 피처 (간단한 이동평균 기반)
    if 'close' in df_window.columns:
        close_prices = df_window['close'].values
        if len(close_prices) >= 20:
            ma5 = np.mean(close_prices[-5:])
            ma20 = np.mean(close_prices[-20:])
            features.extend([
                (close_prices[-1] - ma5) / (ma5 + 1e-8),  # 5일 MA 대비 현재가
                (ma5 - ma20) / (ma20 + 1e-8),  # MA 교차
            ])
        else:
            features.extend([0, 0])
    else:
        features.extend([0, 0])
    
    # 64차원으로 패딩 (앞서 만든 피처들 + 나머지는 0)
    while len(features) < 64:
        features.append(0.0)
    
    return np.array(features[:64], dtype=np.float32)
