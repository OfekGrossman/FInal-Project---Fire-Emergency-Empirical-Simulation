import pandas as pd
import numpy as np
from config import NUM_SAMPLES
from config import EMPERICAL_FILE

def generate_interarrival_samples_by_fdcall(path_file: str):
    """
    Given a CSV of empirical PMFs aggregated by FD_call
    (columns: FD_call, Parameter, Value, Probability),
    return: { parameter: { fd_call_idx: [N samples] } }
    """
    N = NUM_SAMPLES
    df = pd.read_csv(path_file)

    # Map FD_call to 0-based indices
    call_ids = {v: i for i, v in enumerate(sorted(df["FD_call"].unique()))}
    df["FD_call"] = df["FD_call"].map(call_ids)

    out = {}
    for parameter in df["Parameter"].unique():
        d = {}
        sub = df[df["Parameter"] == parameter]
        for fd_call, g in sub.groupby("FD_call"):
            values = g["Value"].values
            probs = g["Probability"].values
            probs = probs / probs.sum()
            samples = np.random.choice(values, size=N, replace=True, p=probs)
            d[fd_call] = list(samples)
        out[parameter] = d
    return out

def generate_empirical_samples_from_pmf(path_file):
    """
    Given a CSV of empirical PMFs (columns: FD_call, FD_response, Parameter, Value, Probability),
    returns a dictionary for each parameter with keys (fd_call_idx, fd_response_idx) as 0-based indices,
    and values as lists of N sampled values according to the empirical probabilities.

    Args:
        pmf_file (str): Path to CSV file.
        N (int): Number of samples to generate per (fd_call, fd_response) combo.

    Returns:
        dict: parameter_dicts[parameter][(fd_call_idx, fd_response_idx)] = [sampled values]
    """
    N = NUM_SAMPLES            # Number of samples you wish to generate (should be defined)

    pmf_df = pd.read_csv(path_file)  # Load PMF data from CSV

    # === Create mappings from FD_call and FD_response values to 0-based indices ===
    call_ids = {v: i for i, v in enumerate(sorted(pmf_df["FD_call"].unique()))}
    resp_ids = {v: i for i, v in enumerate(sorted(pmf_df["FD_response"].unique()))}

    # === Replace FD_call and FD_response values in DataFrame with their new indices ===
    pmf_df["FD_call"] = pmf_df["FD_call"].map(call_ids)
    pmf_df["FD_response"] = pmf_df["FD_response"].map(resp_ids)

    parameter_dicts = {}  # Main output: dict of parameter -> {(fd_call_idx, fd_response_idx): [samples]}

    # === For each parameter (e.g., RESPONSE_TIME_MIN), build the sampling dictionary ===
    for parameter in pmf_df["Parameter"].unique():
        d = {}
        sub_df = pmf_df[pmf_df["Parameter"] == parameter]  # Filter for this parameter
        grouped = sub_df.groupby(["FD_call", "FD_response"])  # Group by each (call, response) pair
        for (fd_call, fd_response), group in grouped:
            values = group["Value"].values                # All possible observed values for this combo
            probs = group["Probability"].values           # Their associated probabilities
            probs = probs / probs.sum()                   # Normalize to ensure they sum to 1
            # === Draw N samples according to the empirical probabilities (replace = True -> the same value can appear multiple times ) ===
            samples = np.random.choice(values, size=N, replace=True, p=probs)
            d[(fd_call, fd_response)] = list(samples)     # Store the samples in the dict with index-based key
        parameter_dicts[parameter] = d  # Store the dict for this parameter

    return parameter_dicts  # The top-level dictionary: parameter -> {(fd_call_idx, fd_response_idx): [samples]}

