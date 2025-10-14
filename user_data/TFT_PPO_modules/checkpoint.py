# user_data/TFT_PPO_Modules/checkpoint.py
import os
import torch

class ModelCheckpoint:
    """
    Monitors validation Sharpe ratio, saves best model, triggers early stopping.
    """

    def __init__(self, save_dir="user_data/models/best", patience=3):
        self.best_metric = -999
        self.patience = patience
        self.counter = 0
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def update(self, model, metric, model_name="ppo_best.zip"):
        if metric > self.best_metric:
            self.best_metric = metric
            model.save(os.path.join(self.save_dir, model_name))
            self.counter = 0
            print(f"✅ New best model saved: {model_name} (Sharpe={metric:.3f})")
        else:
            self.counter += 1
            print(f"⚠️ No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                print("⏹ Early stopping triggered.")
                return True
        return False
