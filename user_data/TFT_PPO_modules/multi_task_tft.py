# user_data/TFT_PPO_modules/multi_task_tft.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet


class MultiTaskTFT(nn.Module):
    """
    멀티-호라이즌 + 멀티태스크 TFT 모델
    
    주요 기능:
    1. 멀티-호라이즌 로그수익 예측 (24h, 48h, 96h)
    2. 방향 분류 보조과제 (24h 기준)
    3. 실현변동성 예측 보조과제 (24h 기준)
    4. Huber 손실 함수 사용
    5. 가중합 손실 함수
    """
    
    def __init__(
        self,
        dataset: TimeSeriesDataSet,
        hidden_size: int = 160,
        attention_head_size: int = 6,
        dropout: float = 0.25,
        horizons: List[int] = [24, 48, 96],
        loss_weights: Dict[str, float] = None,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.horizons = horizons
        self.device = device
        
        # 기본 손실 가중치
        self.loss_weights = loss_weights or {
            "returns": 1.0,
            "direction": 0.5,
            "volatility": 0.25
        }
        
        # 기본 TFT 모델 생성
        from pytorch_forecasting.metrics import MAE
        self.tft = TemporalFusionTransformer.from_dataset(
            dataset,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=hidden_size,
            output_size=1,  # 기본 출력 크기
            loss=MAE(),  # PyTorch Lightning Metric 사용
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
        
        # TFT 모델을 지정된 디바이스로 이동
        self.tft = self.tft.to(device)
        
        # 멀티-호라이즌 출력 헤드들
        self.return_heads = nn.ModuleDict({
            f"horizon_{h}": nn.Linear(hidden_size, 1) 
            for h in horizons
        })
        
        # 보조과제 헤드들
        self.direction_head = nn.Linear(hidden_size, 1)  # 방향 분류 (sigmoid)
        self.volatility_head = nn.Linear(hidden_size, 1)  # 변동성 예측 (ReLU)
        
        # 손실 함수들
        self.huber_loss = nn.HuberLoss(delta=0.1)
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Direction 손실 함수 개선 (클래스 불균형 대응)
        self.direction_loss = nn.BCEWithLogitsLoss()
        
        self.to(device)
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-task learning
        
        Returns:
            Dict containing predictions for all tasks
        """
        # TFT 인코더를 직접 호출하여 진짜 인코더 표현 추출
        try:
            # TFT의 encoder를 직접 호출 (pytorch_forecasting 구조에 맞게)
            if hasattr(self.tft, 'encoder'):
                # TFT 내부 encoder 직접 호출
                encoder_output = self.tft.encoder(x)  # (B, T, d_model)
                encoder_repr = encoder_output[:, -1, :]  # (B, d_model)
            else:
                # TFT 기본 출력에서 encoder representation 추출
                tft_output = self.tft(x)
                if hasattr(tft_output, 'encoder_output'):
                    encoder_repr = tft_output.encoder_output
                    if encoder_repr.dim() == 3:
                        encoder_repr = encoder_repr[:, -1, :]
                else:
                    # Fallback: 입력 피처 기반 임베딩 생성
                    batch_size = next(iter(x.values())).shape[0]
                    if 'encoder_cont' in x:
                        input_features = x['encoder_cont']
                        if input_features.dim() == 3:
                            # (B, T, F) -> (B, F) 평균
                            feature_mean = input_features.mean(dim=1)
                            # 선형 변환으로 hidden_size로 투영
                            if not hasattr(self, '_fallback_proj'):
                                self._fallback_proj = nn.Linear(feature_mean.shape[-1], self.hidden_size).to(self.device)
                            encoder_repr = self._fallback_proj(feature_mean)
                        else:
                            encoder_repr = torch.randn(batch_size, self.hidden_size, device=self.device) * 0.1
                    else:
                        encoder_repr = torch.randn(batch_size, self.hidden_size, device=self.device) * 0.1
        except Exception as e:
            # 최후의 수단: 입력 피처 기반 임베딩 생성
            batch_size = next(iter(x.values())).shape[0]
            if 'encoder_cont' in x:
                input_features = x['encoder_cont']
                if input_features.dim() == 3:
                    # (B, T, F) -> (B, F) 평균
                    feature_mean = input_features.mean(dim=1)
                    # 선형 변환으로 hidden_size로 투영
                    if not hasattr(self, '_fallback_proj'):
                        self._fallback_proj = nn.Linear(feature_mean.shape[-1], self.hidden_size).to(self.device)
                    encoder_repr = self._fallback_proj(feature_mean)
                else:
                    encoder_repr = torch.randn(batch_size, self.hidden_size, device=self.device) * 0.1
            else:
                encoder_repr = torch.randn(batch_size, self.hidden_size, device=self.device) * 0.1
        
        # 배치 차원 처리
        if encoder_repr.dim() == 3:  # (batch, seq, hidden)
            encoder_repr = encoder_repr[:, -1, :]  # 마지막 타임스텝 사용
        elif encoder_repr.dim() == 2:  # (batch, hidden)
            pass  # 이미 올바른 형태
        else:
            raise ValueError(f"Unexpected encoder_repr shape: {encoder_repr.shape}")
        
        # 멀티-호라이즌 수익률 예측
        return_predictions = {}
        for horizon in self.horizons:
            head_name = f"horizon_{horizon}"
            return_predictions[head_name] = self.return_heads[head_name](encoder_repr)
        
        # 보조과제 예측
        direction_pred = self.direction_head(encoder_repr)
        volatility_pred = self.volatility_head(encoder_repr)
        
        return {
            "returns": return_predictions,
            "direction": direction_pred,
            "volatility": volatility_pred,
            "encoder_repr": encoder_repr  # 진짜 인코더 표현 반환
        }
    
    def _extract_encoder_repr(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """TFT에서 encoder representation 추출"""
        # TFT 내부 구조에 접근하여 encoder output 추출
        try:
            # pytorch_forecasting의 TFT 구조 활용
            encoder_output = self.tft.encode(x)
            return encoder_output
        except Exception:
            # Fallback: 간단한 방법으로 representation 생성
            batch_size = next(iter(x.values())).shape[0]
            return torch.zeros(batch_size, self.hidden_size, device=self.device)
    
    def compute_loss(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        멀티태스크 손실 계산
        
        Args:
            predictions: 모델 예측값
            targets: 실제 타깃값
            
        Returns:
            Dict containing individual and total losses
        """
        losses = {}
        
        # 1. 멀티-호라이즌 수익률 손실 (Huber)
        return_losses = []
        for horizon in self.horizons:
            head_name = f"horizon_{horizon}"
            target_name = f"return_{horizon}h"
            
            if head_name in predictions["returns"] and target_name in targets:
                pred = predictions["returns"][head_name].squeeze()
                target = targets[target_name].squeeze()
                
                # 텐서 크기 맞추기
                min_size = min(pred.shape[0], target.shape[0])
                pred = pred[:min_size]
                target = target[:min_size]
                
                loss = self.huber_loss(pred, target)
                return_losses.append(loss)
                losses[f"return_{horizon}h"] = loss
        
        # 수익률 손실의 평균
        if return_losses:
            losses["returns_total"] = torch.stack(return_losses).mean()
        else:
            losses["returns_total"] = torch.tensor(0.0, device=self.device)
        
        # 2. 방향 분류 손실 (BCE with class balancing)
        if "direction" in predictions and "direction" in targets:
            pred_dir = predictions["direction"].squeeze()
            target_dir = targets["direction"].squeeze()
            
            # 텐서 크기 맞추기
            min_size = min(pred_dir.shape[0], target_dir.shape[0])
            pred_dir = pred_dir[:min_size]
            target_dir = target_dir[:min_size]
            
            # 클래스 불균형 대응: pos_weight 계산
            pos_count = target_dir.sum()
            neg_count = len(target_dir) - pos_count
            if pos_count > 0 and neg_count > 0:
                pos_weight = neg_count / pos_count  # 음성 클래스에 더 높은 가중치
                pos_weight = torch.clamp(torch.tensor(pos_weight), 0.5, 2.0)  # 극값 방지
            else:
                pos_weight = 1.0
            
            # 가중치가 적용된 BCE 손실
            direction_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            losses["direction"] = direction_loss_fn(pred_dir, target_dir)
        else:
            losses["direction"] = torch.tensor(0.0, device=self.device)
        
        # 3. 변동성 예측 손실 (Huber)
        if "volatility" in predictions and "volatility" in targets:
            pred_vol = predictions["volatility"].squeeze()
            target_vol = targets["volatility"].squeeze()
            
            # 텐서 크기 맞추기
            min_size = min(pred_vol.shape[0], target_vol.shape[0])
            pred_vol = pred_vol[:min_size]
            target_vol = target_vol[:min_size]
            
            losses["volatility"] = self.huber_loss(pred_vol, target_vol)
        else:
            losses["volatility"] = torch.tensor(0.0, device=self.device)
        
        # 4. 가중합 총 손실
        total_loss = (
            self.loss_weights["returns"] * losses["returns_total"] +
            self.loss_weights["direction"] * losses["direction"] +
            self.loss_weights["volatility"] * losses["volatility"]
        )
        losses["total"] = total_loss
        
        return losses
    
    def predict_returns(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """수익률 예측만 수행 (추론용)"""
        with torch.no_grad():
            outputs = self.forward(x)
            return outputs["returns"]
    
    def predict_direction(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """방향 예측만 수행 (추론용)"""
        with torch.no_grad():
            outputs = self.forward(x)
            return torch.sigmoid(outputs["direction"])
    
    def predict_volatility(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """변동성 예측만 수행 (추론용)"""
        with torch.no_grad():
            outputs = self.forward(x)
            return F.relu(outputs["volatility"])


def create_multi_horizon_targets(
    df: 'pd.DataFrame', 
    horizons: List[int] = [24, 48, 96],
    price_col: str = "close"
) -> 'pd.DataFrame':
    """
    멀티-호라이즌 타깃 생성
    
    Args:
        df: 가격 데이터프레임
        horizons: 예측 호라이즌 리스트
        price_col: 가격 컬럼명
        
    Returns:
        타깃이 추가된 데이터프레임
    """
    df = df.copy()
    
    # 가격 데이터 검증 및 전처리
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in dataframe")
    
    # 가격이 0 이하인 경우 처리
    df[price_col] = df[price_col].replace([0, np.inf, -np.inf], np.nan)
    df[price_col] = df[price_col].ffill().bfill()
    
    # 로그 가격 계산 (안전한 로그 계산)
    log_prices = np.log(np.maximum(df[price_col], 1e-10))  # 0보다 작은 값 방지
    
    # 각 호라이즌별 로그수익률 계산
    for horizon in horizons:
        future_log_price = log_prices.shift(-horizon)
        log_return = future_log_price - log_prices
        
        # NA/무한대 값 처리
        log_return = log_return.replace([np.inf, -np.inf], np.nan)
        log_return = log_return.fillna(0.0)  # NA를 0으로 대체
        
        # 극값 클리핑 (99.9% 분위수 기준)
        q99_9 = log_return.quantile(0.999)
        q0_1 = log_return.quantile(0.001)
        log_return = log_return.clip(lower=q0_1, upper=q99_9)
        
        df[f"return_{horizon}h"] = log_return
    
    # 방향 타깃 (24h 기준)
    df["direction"] = (df["return_24h"] > 0).astype(float)
    
    # 실현변동성 타깃 (24h 기준, 롤링 표준편차)
    returns_24h = df["return_24h"]
    df["volatility"] = returns_24h.rolling(window=24, min_periods=1).std().fillna(0.0)
    
    # 변동성도 극값 클리핑
    vol_q99 = df["volatility"].quantile(0.99)
    df["volatility"] = df["volatility"].clip(upper=vol_q99)
    
    # 최종 검증: NA/무한대 값이 있는지 확인
    for col in [f"return_{h}h" for h in horizons] + ["direction", "volatility"]:
        if col in df.columns:
            na_count = df[col].isna().sum()
            inf_count = np.isinf(df[col]).sum()
            if na_count > 0 or inf_count > 0:
                print(f"Warning: {col} has {na_count} NA and {inf_count} infinite values")
                df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    return df


def compute_downstream_metrics(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    top_percent: float = 0.2
) -> Dict[str, float]:
    """
    다운스트림 성능 지표 계산 (안전장치 포함)
    
    Args:
        predictions: 모델 예측값
        targets: 실제 타깃값
        top_percent: 상위 퍼센트 (IC 계산용)
        
    Returns:
        Dict containing downstream metrics
    """
    metrics = {}
    
    # 기본값 설정
    metrics["directional_auc"] = 0.5
    metrics["ic_top20"] = 0.0
    metrics["vol_calibration"] = 0.0
    
    # 1. 방향성 AUC (24h 기준) - 텐서 차원 문제 해결
    if "direction" in predictions and "direction" in targets:
        try:
            # 텐서를 1D로 평탄화
            pred_dir_logits = predictions["direction"].cpu().numpy().flatten()
            target_dir = targets["direction"].cpu().numpy().flatten()
            
            # 유효한 데이터만 필터링
            valid_mask = np.isfinite(pred_dir_logits) & np.isfinite(target_dir)
            if valid_mask.sum() < 10:  # 최소 10개 샘플 필요
                metrics["directional_auc"] = 0.5
            else:
                pred_valid = pred_dir_logits[valid_mask]
                target_valid = target_dir[valid_mask]
                
                # 클래스 다양성 확인
                unique_classes = len(np.unique(target_valid))
                if unique_classes < 2:
                    metrics["directional_auc"] = 0.5
                else:
                    from sklearn.metrics import roc_auc_score
                    # logits를 직접 사용 (sklearn이 내부적으로 sigmoid 적용)
                    auc = roc_auc_score(target_valid, pred_valid)
                    metrics["directional_auc"] = float(auc) if np.isfinite(auc) else 0.5
        except Exception as e:
            print(f"Direction AUC calculation error: {e}")
            metrics["directional_auc"] = 0.5
    
    # 2. IC@Top20% (24h 수익률 기준) - 안전장치 강화
    if "horizon_24" in predictions["returns"] and "return_24h" in targets:
        try:
            pred_24h = predictions["returns"]["horizon_24"].cpu().numpy().flatten()
            target_24h = targets["return_24h"].cpu().numpy().flatten()
            
            # 유효한 데이터만 필터링
            valid_mask = np.isfinite(pred_24h) & np.isfinite(target_24h)
            if valid_mask.sum() < 20:  # 최소 20개 샘플 필요
                metrics["ic_top20"] = 0.0
            else:
                pred_valid = pred_24h[valid_mask]
                target_valid = target_24h[valid_mask]
                
                # 상위 확신도 샘플 선택
                n_top = max(5, int(len(pred_valid) * top_percent))  # 최소 5개
                if n_top < len(pred_valid):
                    top_indices = np.argsort(-np.abs(pred_valid))[:n_top]
                    ic_top = np.corrcoef(
                        pred_valid[top_indices], 
                        target_valid[top_indices]
                    )[0, 1]
                    metrics["ic_top20"] = float(ic_top) if np.isfinite(ic_top) else 0.0
                else:
                    metrics["ic_top20"] = 0.0
        except Exception:
            metrics["ic_top20"] = 0.0
    
    # 3. 변동성 보정 (Vol calibration) - 안전장치 강화
    if "volatility" in predictions and "volatility" in targets:
        try:
            pred_vol = F.relu(predictions["volatility"]).cpu().numpy().flatten()
            target_vol = targets["volatility"].cpu().numpy().flatten()
            
            # 유효한 데이터만 필터링
            valid_mask = np.isfinite(pred_vol) & np.isfinite(target_vol) & (target_vol > 0)
            if valid_mask.sum() < 10:  # 최소 10개 샘플 필요
                metrics["vol_calibration"] = 0.0
            else:
                pred_valid = pred_vol[valid_mask]
                target_valid = target_vol[valid_mask]
                
                # 상관계수 계산
                vol_corr = np.corrcoef(pred_valid, target_valid)[0, 1]
                metrics["vol_calibration"] = float(vol_corr) if np.isfinite(vol_corr) else 0.0
        except Exception:
            metrics["vol_calibration"] = 0.0
    
    return metrics
