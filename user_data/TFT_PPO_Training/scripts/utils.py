# user_data/TFT_PPO_Training/scripts/utils.py
import os
import torch
import random
import numpy as np


def setup_device(gpu_id: int | None = None, verbose: bool = True) -> torch.device:
    """
    Selects and configures computation device for training/inference.

    Parameters
    ----------
    gpu_id : int | None
        Specific GPU index to use (e.g., 0). If None, uses the first available GPU.
    verbose : bool
        If True, prints device info.

    Returns
    -------
    torch.device
        Configured device object.
    """
    if torch.cuda.is_available():
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device("cuda")
        if verbose:
            print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        if verbose:
            print("Using CPU mode.")
    return device


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set all relevant random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Base random seed.
    deterministic : bool
        If True, enables deterministic algorithms (may reduce performance slightly).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Optional: deterministic cuDNN behavior for full reproducibility
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    print(f"Random seed set to {seed} | Deterministic={deterministic}")


def ensure_dir(path: str, verbose: bool = False) -> str:
    """
    Ensure directory exists; create if missing.

    Parameters
    ----------
    path : str
        Target directory path.
    verbose : bool
        If True, prints confirmation message.

    Returns
    -------
    str
        The validated absolute path.
    """
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    if verbose:
        print(f"Directory ready: {abs_path}")
    return abs_path
