import json
import platform
import subprocess
from pathlib import Path
import time
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib

rc = matplotlib.rcParams


def set_plot_style():
    small_font = 5
    medium_font = 6
    big_font = 7
    rc["figure.dpi"] = 300
    rc["font.size"] = small_font
    rc["axes.titlesize"] = medium_font  # title of each sub-figure
    rc["axes.labelsize"] = medium_font
    rc["xtick.labelsize"] = small_font
    rc["ytick.labelsize"] = small_font
    rc["legend.fontsize"] = small_font
    rc["figure.titlesize"] = big_font  # title of the whole figure
    rc["font.family"] = "sans-serif"
    rc["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    rc["mathtext.fontset"] = "custom"
    rc["mathtext.rm"] = "Arial"
    rc["mathtext.it"] = "Arial:italic"
    rc["mathtext.bf"] = "Arial:bold"
    rc["axes.linewidth"] = 0.4
    rc["xtick.major.width"] = 0.4
    rc["ytick.major.width"] = 0.4
    rc["xtick.minor.width"] = 0.3
    rc["ytick.minor.width"] = 0.3


def convert_event_to_dataset(
    x_arr: NDArray,
    t_arr: NDArray,
    y_matrix: NDArray,
    q_matrix: NDArray,
    look_back: int,
    pipe_params: dict,
) -> tuple[NDArray, NDArray]:
    """Feature: [y1(t-T:t+1), y2(t-T:t+1), d, s, l, n, x]
    Labels: [y, q, dy/dx, dq/dx]

    Args:
        x_arr (NDArray): _description_
        t_arr (NDArray): _description_
        y_matrix (NDArray): _description_
        q_matrix (NDArray): _description_
        look_back (int): _description_
        pipe_params (dict): _description_

    Returns:
        tuple[NDArray, NDArray]: _description_
    """
    # assert (
    #     isinstance(lookBack, int)
    # ), "This function assume the interval of each sensor record is 1 second! Otherwise, we need more complex method to get item from record."

    # extract data from event
    features = []
    labels = []
    for i in range(look_back, y_matrix.shape[0]):
        feature_branch = np.hstack(
            [
                y_matrix[i - look_back : i + 1, 0],  # upstream BC: water depth
                # q_matrix[i - look_back : i + 1, 0], # upstream BC: inflow discharge
                y_matrix[i - look_back : i + 1, -1],  # downstream BC: water depth
                # np.array([pipe_params["d"]]),   # do not need diameter
                np.array([pipe_params["s"]]),
                np.array([pipe_params["l"]]),
                np.array([pipe_params["n"]]),
            ]
        )
        # shape: (2 * (T + 1) + 4) = (16,) for T = 5
        matrix_2d = np.tile(feature_branch, (len(x_arr), 1))
        feature_matrix = np.hstack([matrix_2d, x_arr.reshape(-1, 1)])
        # shape: (2 * (T + 1) + 4 + 1) = (L + 2, 17) for T = 5

        label_matrix = np.vstack(
            [
                y_matrix[i, :],
                q_matrix[i, :],
            ]
        ).T  # shape: (L + 2, 2 + 2)
        assert feature_matrix.shape[0] == label_matrix.shape[0]
        features.append(feature_matrix)
        labels.append(label_matrix)  # shape: ?

    # convert data type
    features = np.vstack(features)  # shape: (N * 17, 2)
    labels = np.vstack(labels)  # shape: (N * (L+2), 2)

    return features, labels


def get_metrics(y1: np.ndarray, y2: np.ndarray) -> tuple[float, float, float]:
    r2 = R_squared(y1, y2)
    rmse = root_mean_squared_error(y1, y2)
    mre = relative_error(y1, y2)
    return r2, rmse, mre


def R_squared(y_true_arr: np.ndarray, y_est_arr: np.ndarray) -> float:
    mean = y_true_arr.mean()
    nominator = ((y_true_arr - y_est_arr) ** 2).sum()
    denominator = ((y_true_arr - mean) ** 2).sum()
    return 1 - nominator / denominator


def mean_squared_error(y_true_arr: np.ndarray, y_est_arr: np.ndarray) -> float:
    # mean_squared_error
    y1 = y_true_arr
    y2 = y_est_arr
    return ((y1 - y2) ** 2).sum() / (y1.size)


def root_mean_squared_error(y_true_arr: np.ndarray, y_est_arr: np.ndarray) -> float:
    # root mean_squared_error
    return np.sqrt(mean_squared_error(y_true_arr, y_est_arr))


def relative_error(y_true_arr: np.ndarray, y_est_arr: np.ndarray) -> float:
    error = np.sum(np.abs(y_true_arr - y_est_arr) / y_true_arr) / y_true_arr.size
    return error


def calculate_metric_table(
    y_true_arr: NDArray,
    y_pred_arr: NDArray,
) -> pd.DataFrame:
    df = pd.DataFrame(columns=["R2", "MRE", "RMSE"])
    idx = 0
    y1_arr = y_true_arr[..., idx].flatten()
    y2_arr = y_pred_arr[..., idx].flatten()
    r2, rmse, mre = get_metrics(y1_arr, y2_arr)
    df.loc["y"] = [r2, mre, rmse]
    idx = 1
    y1_arr = y_true_arr[..., idx].flatten()
    y2_arr = y_pred_arr[..., idx].flatten()
    r2, rmse, mre = get_metrics(y1_arr, y2_arr)
    df.loc["q"] = [r2, mre, rmse]
    return df


import torch
from torch import Tensor
from operators import MultiExpertNeuralOperator


def print_device_specs(device: str | torch.device) -> None:
    """Print the CPU model and selected GPU specifications."""
    device = torch.device(device)
    print(f"Using {device} device")
    cpu_model = platform.processor() or platform.machine()
    if Path("/proc/cpuinfo").exists():
        cpu_model = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu_model,
        )
    elif platform.system() == "Darwin":
        try:
            cpu_model = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    print(f"CPU: {cpu_model}")
    if device.type == "cuda":
        gpu = torch.cuda.get_device_properties(device)
        print(f"GPU: {gpu.name} ({gpu.total_memory / 1024**3:.1f} GiB VRAM)")
    elif device.type == "mps":
        gpu_model = "Apple GPU"
        try:
            displays = json.loads(
                subprocess.check_output(
                    ["system_profiler", "SPDisplaysDataType", "-json"], text=True
                )
            )
            gpu_model = displays["SPDisplaysDataType"][0]["sppci_model"]
        except (OSError, subprocess.CalledProcessError, ValueError, KeyError, IndexError):
            pass
        print(f"GPU: {gpu_model} (shared memory)")
    else:
        print("GPU: not used (CPU execution)")


class Database:
    def __init__(self, filepath: str):
        loaded_sve_data = np.load(filepath, allow_pickle=True)
        self.t_arr = loaded_sve_data["time"]
        self.pipe_data = loaded_sve_data["pipe"].item()
        self.outfall_data = loaded_sve_data["outfall"].item()
        self.manhole_data = loaded_sve_data["manhole"].item()

    def extract_pipe_dataset(
        self,
        cdt_name: str,
        LOOK_BACK: int = 5,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        variables = self.pipe_data[cdt_name]
        # Extract variables
        pipe_params = {
            "s": variables["s"],
            "l": variables["l"],
            "n": variables["n"],
        }
        x_arr = variables["x"]
        t_arr = variables["t"]
        y_matrix = variables["y"]
        q_matrix = variables["q"]

        x_arr = x_arr.astype(np.float32)
        t_arr = t_arr.astype(np.float32)
        y_matrix = y_matrix.astype(np.float32)
        q_matrix = q_matrix.astype(np.float32)
        feature_arr, label_arr = convert_event_to_dataset(
            x_arr,
            t_arr,
            y_matrix,
            q_matrix,
            LOOK_BACK,
            pipe_params,
        )
        return x_arr, t_arr, feature_arr, label_arr

    def extract_dataset_for_diameter(
        self,
        diameter: int,
        device: str,
        LOOK_BACK: int = 5,
    ) -> tuple[int, Tensor, Tensor]:
        count = 0
        feature_arr_list, Y_true_arr_list = [], []
        for name, variables in self.pipe_data.items():
            if variables["d"] != diameter:
                continue
            count += 1
            x_arr, t_arr, feature_arr, Y_true_arr = self.extract_pipe_dataset(name, LOOK_BACK)
            feature_arr_list.append(feature_arr)
            Y_true_arr_list.append(Y_true_arr)

        if len(feature_arr_list) > 0:
            feature_arr_all = np.concatenate(feature_arr_list, axis=0)
            Y_true_arr_all = np.concatenate(Y_true_arr_list, axis=0)
            features, labels = numpy_to_tensor(feature_arr_all, Y_true_arr_all, device)
        else:
            features, labels = torch.tensor([]), torch.tensor([])
        return count, features, labels


def numpy_to_tensor(
    feature_arr: NDArray, label_arr: NDArray, device: str
) -> tuple[Tensor, Tensor]:
    features = torch.from_numpy(feature_arr).float()
    labels = torch.from_numpy(label_arr).float()
    features = features.to(device)
    labels = labels.to(device)
    return features, labels


def predict_pipe(
    db: Database, cdt_name: str, surrogate: MultiExpertNeuralOperator, device: str
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    variables: dict = db.pipe_data[cdt_name]  # load a pipe
    x_arr, t_arr, feature_arr, Y_true_arr = db.extract_pipe_dataset(cdt_name)
    features, labels = numpy_to_tensor(feature_arr, Y_true_arr, device)
    model = surrogate.model_dict[variables["d"]]
    model.eval()
    with torch.no_grad():
        Y_pred = model.predict(features)
    Y_pred_arr = Y_pred.cpu().numpy()
    return x_arr, t_arr, Y_pred_arr, Y_true_arr


def parse_prediction(
    Y_pred_arr: NDArray,
    Y_true_arr: NDArray,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    y_pred_arr = Y_pred_arr[:, 0]
    q_pred_arr = Y_pred_arr[:, 1]
    y_true_arr = Y_true_arr[:, 0]
    q_true_arr = Y_true_arr[:, 1]
    return y_pred_arr, q_pred_arr, y_true_arr, q_true_arr


def get_metrics_table(
    Y_true_arr: NDArray,
    Y_pred_arr: NDArray,
    data_dict: dict[int, tuple[int, Tensor, Tensor]],
) -> pd.DataFrame:
    """Report R² and NMAE using diameter and full-pipe Manning capacity.

    Arrays must follow the same diameter/sample order as data_dict.
    Diameter keys are in millimetres; depth and flow are in metres and m³/s.
    """
    scales = []
    for diameter, (count, features, labels) in data_dict.items():
        if count == 0:
            continue
        feature_arr = features.detach().cpu().numpy()
        D = diameter / 1000
        slope = feature_arr[:, -4]
        roughness = feature_arr[:, -2]
        q_max = np.pi * (D / 2) ** 2 * (D / 4) ** (2 / 3) * np.sqrt(slope) / roughness
        scales.append(np.column_stack((np.full(len(feature_arr), D), q_max)))
    if not scales:
        raise ValueError("No samples available for the metrics table")
    scale_arr = np.concatenate(scales, axis=0)
    if Y_true_arr.shape != Y_pred_arr.shape or Y_true_arr.shape != scale_arr.shape:
        raise ValueError("Observations, predictions, and normalization scales must align")
    if not np.all(np.isfinite(scale_arr) & (scale_arr > 0)):
        raise ValueError("NMAE requires finite, positive diameters and Manning capacities")
    df = pd.DataFrame(columns=["R2", "NMAE"])
    for idx, variable in enumerate(("y", "q")):
        observed = Y_true_arr[:, idx]
        predicted = Y_pred_arr[:, idx]
        nmae = np.mean(np.abs(observed - predicted) / scale_arr[:, idx])
        df.loc[variable] = [R_squared(observed, predicted), f"{nmae * 100:.4f}%"]
    return df


def get_domain(nx: int, Y_pred_arr: NDArray, Y_true_arr: NDArray) -> tuple[NDArray, NDArray]:
    n, _ = Y_pred_arr.shape
    nt = int(n / nx)  # number of time steps
    Y_pred_domain_arr = Y_pred_arr.reshape(nt, nx, 2)
    Y_true_domain_arr = Y_true_arr.reshape(nt, nx, 2)
    return Y_pred_domain_arr, Y_true_domain_arr


def concat_dict(dt: dict[int, NDArray]) -> NDArray:
    arr_list = [arr for diameter, arr in dt.items()]
    arr_all = np.concatenate(arr_list, axis=0)
    return arr_all


def concat_data_dict(data_dict: dict[int, NDArray]) -> tuple[NDArray, NDArray]:
    feature_list, label_list = [], []
    for diameter, (count, features, labels) in data_dict.items():
        if count == 0:
            continue
        feature_list.append(features.cpu().numpy())
        label_list.append(labels.cpu().numpy())

    feature_all = np.concatenate(feature_list, axis=0)
    label_all = np.concatenate(label_list, axis=0)
    return feature_all, label_all


def predict_all_network(
    surrogate: MultiExpertNeuralOperator,
    data_dict: dict[int, Tensor],
) -> tuple[dict[int, NDArray], dict[int, NDArray]]:
    label_pred_dict = {}
    label_true_dict = {}
    for diameter, (count, features, labels) in data_dict.items():
        if count == 0:
            continue
        with torch.no_grad():
            Y_pred = surrogate.predict_batch(diameter, features)
        label_pred_dict[diameter] = Y_pred.detach().cpu().numpy()
        label_true_dict[diameter] = labels.detach().cpu().numpy()
    return label_pred_dict, label_true_dict


def predict_with_timing(
    surrogate: MultiExpertNeuralOperator,
    data_dict: dict[int, Tensor],
    warmup: int = 5,
) -> float:
    # Warm-up to avoid one-time kernel/init overhead in timing
    for diameter, (count, features, labels) in data_dict.items():
        if count == 0:
            continue
        with torch.no_grad():
            for _ in range(warmup):
                _ = surrogate.predict_batch(diameter, features)

    if list(data_dict.values())[0][1].is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for diameter, (count, features, labels) in data_dict.items():
        if count == 0:
            continue
        with torch.no_grad():
            Y_pred = surrogate.predict_batch(diameter, features)

    if list(data_dict.values())[0][1].is_cuda:
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    return dt
