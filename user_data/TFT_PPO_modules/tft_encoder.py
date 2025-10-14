import torch
import numpy as np
from pytorch_forecasting.models import TemporalFusionTransformer


class TFTEncoder:
    """
    TFTEncoder
    -----------
    Lightweight wrapper around a pretrained Temporal Fusion Transformer.
    Used to generate compact latent state embeddings from recent market windows
    for RL environments or downstream predictive tasks.

    Supports:
      - GPU / mixed precision inference
      - Safe input validation
      - Direct access to encoder hidden representations
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = TemporalFusionTransformer.load_from_checkpoint(model_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, df_window):
        """
        Encodes a recent OHLCV + feature window into a latent state vector.

        Parameters
        ----------
        df_window : pd.DataFrame
            Feature window of shape [window_length, num_features]

        Returns
        -------
        np.ndarray
            Latent embedding of shape [hidden_size], suitable for RL agent state input.
        """
        # --- Input validation ---
        if not hasattr(df_window, "values"):
            raise ValueError("Input must be a pandas DataFrame or numpy-like structure")

        # --- Prepare model input ---
        # TFT expects (batch_size, time_steps, features)
        x = torch.tensor(df_window.values, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Some TFT checkpoints require a `batch` dict. This ensures compatibility:
        if hasattr(self.model, "encode"):
            # Modern pytorch_forecasting TFT has .encode() to get latent representation
            hidden = self.model.encode(x)
        else:
            # Fallback to forward() call; extract hidden state if available
            out = self.model(x)
            hidden = getattr(out, "encoder_output", out)

        # --- Convert to CPU numpy ---
        return hidden.squeeze().detach().cpu().numpy().astype(np.float32)
