import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from torch.utils.data import DataLoader
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from TFT_PPO_modules.feature_pipeline import FeaturePipeline
from TFT_PPO_modules.trading_env import TradingEnv

# =========================
# 1. 데이터 로드 및 분리
# =========================
print("Loading data...")

df = pd.read_csv("user_data/data/BTC_USDT-1h.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
df['log_return'] = np.log(df['close'] / df['close'].shift(1))
df = df.dropna().reset_index(drop=True)

# 분리: TFT = 5년 / PPO = 최근 6개월
cut_date = df['date'].max() - pd.Timedelta(days=180)
df_tft = df.copy()
df_ppo = df[df['date'] >= cut_date].reset_index(drop=True)

# =========================
# 2. Feature Engineering
# =========================
print("Generating features...")
fp = FeaturePipeline()
df_tft = fp.add_features(df_tft)
df_ppo = fp.add_features(df_ppo)

df_tft['log_return'] = np.log(df_tft['close'] / df_tft['close'].shift(1))
df_tft = df_tft.dropna().reset_index(drop=True)

# =========================
# 3. TFT 학습 (5년치)
# =========================
print("Training TFT Encoder (5-year data)...")

df_tft['time_idx'] = (df_tft['date'] - df_tft['date'].min()).dt.total_seconds() // 3600
df_tft['group_id'] = 'BTCUSDT'

max_encoder_length = 72

tft_dataset = TimeSeriesDataSet(
    df_tft,
    time_idx='time_idx',
    target='log_return',
    group_ids=['group_id'],
    time_varying_known_reals=[],
    time_varying_unknown_reals=fp.features,
    min_encoder_length=max_encoder_length,
    max_encoder_length=max_encoder_length,
    target_normalizer=NaNLabelEncoder(),
)

train_loader = tft_dataset.to_dataloader(train=True, batch_size=64, num_workers=0)

tft = TemporalFusionTransformer.from_dataset(
    tft_dataset,
    learning_rate=1e-4,
    hidden_size=64,
    attention_head_size=4,
    dropout=0.1,
    output_size=1,
    loss=torch.nn.L1Loss(),
)

optimizer = torch.optim.Adam(tft.parameters(), lr=1e-4)

tft.train()
for epoch in range(10):
    losses = []
    for x, y in train_loader:
        optimizer.zero_grad()
        out = tft(x)
        loss = torch.mean(torch.abs(out - y))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f"[TFT EPOCH {epoch+1}] Loss: {np.mean(losses):.6f}")

# 모델 저장
torch.save(tft.state_dict(), "user_data/models/tft_encoder.pt")
print("TFT Encoder saved to models/")

# =========================
# 4. PPO 학습 (최근 6개월)
# =========================
print("Training PPO (last 6 months)...")

tft.eval()
env = DummyVecEnv([lambda: TradingEnv(df_ppo, tft_model=tft, features=fp.features)])

ppo_model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-5,
    n_steps=2048,
    batch_size=512,
    gamma=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
)

ppo_model.learn(total_timesteps=2_000_000)
ppo_model.save("user_data/models/ppo_policy.zip")
print("PPO model saved to models/")

print("All training complete! Ready for Freqtrade strategy.")
