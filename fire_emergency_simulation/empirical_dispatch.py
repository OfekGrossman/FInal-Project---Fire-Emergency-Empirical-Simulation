import pandas as pd
import numpy as np

def prepare_empirical_dispatch(dispatch_csv):
    """
    Loads and remaps FD_call and vehicle IDs to 0-based indices.
    Returns:
        dispatch_dict: {(fd_call_idx, available_set_str): ([response_indices], [probabilities])}
    """
    df = pd.read_csv(dispatch_csv)

    # --- Build mappings ---
    all_vehicles = set(df["FD_response"].astype(str).unique())
    for aset in df["available_set"]:
        for v in str(aset).split(","):
            all_vehicles.add(v.strip())
    all_vehicles = sorted(all_vehicles)
    fd_response_map = {v: i for i, v in enumerate(all_vehicles)}

    all_calls = sorted(df["FD_call"].astype(str).unique())
    fd_call_map = {c: i for i, c in enumerate(all_calls)}

    # --- Remap DataFrame ---
    df["FD_call"] = df["FD_call"].astype(str).map(fd_call_map)
    df["FD_response"] = df["FD_response"].astype(str).map(fd_response_map)

    def remap_available_set(aset):
        vehicles = [v.strip() for v in str(aset).split(",")]
        indices = [str(fd_response_map[v]) for v in vehicles]
        return ",".join(sorted(indices))
    df["available_set"] = df["available_set"].apply(remap_available_set)

    # --- Build the dispatch dictionary ---
    dispatch_dict = {}
    for (fd_call, available_set), group in df.groupby(["FD_call", "available_set"]):
        responses = group["FD_response"].tolist()
        probs = group["probability"].values
        probs = probs / probs.sum()
        dispatch_dict[(str(fd_call), str(available_set))] = (responses, probs)

    return dispatch_dict

def empirical_dispatch(dispatch_dict, fd_call, available_vehicles):
    """
    Dispatches a vehicle based on the empirical distribution.
    Args:
        dispatch_dict: lookup dict from prepare_empirical_dispatch
        fd_call: index (0-based)
        available_vehicles: list of indices (0-based)
    Returns:
        index of the dispatched vehicle (int, 0-4)
    """
    available_set_str = ",".join(str(x) for x in sorted(available_vehicles))
    key = (str(fd_call), available_set_str)
    if key not in dispatch_dict:
        raise ValueError(f"No empirical dispatch data for FD_call={fd_call} and available_set={available_set_str}")
    responses, probs = dispatch_dict[key]
    return int(np.random.choice(responses, p=probs))
