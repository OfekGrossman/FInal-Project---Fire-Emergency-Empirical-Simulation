import pandas as pd
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_dispatch_histograms(dispatch_df, output_folder="dispatch_histograms"):
    """
    Generates and saves a histogram for each (FD_call, available_set) combination,
    showing the empirical dispatch distribution over vehicles.

    Args:
        dispatch_df: DataFrame from build_empirical_dispatch_distribution().
        output_folder: Directory where the histograms will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)
    grouped = dispatch_df.groupby(["FD_call", "available_set"])
    for (fd_call, available_set), group in grouped:
        plt.figure(figsize=(8, 5))
        vehicles = group["FD_response"].astype(str)
        counts = group["count"]
        probs = group["probability"]

        plt.bar(vehicles, probs, color="steelblue", edgecolor="black")
        plt.title(f"FD_call: {fd_call} | Available: [{available_set}]")
        plt.xlabel("FD_response (Vehicle)")
        plt.ylabel("Empirical Dispatch Probability")
        plt.ylim(0, 1)
        plt.tight_layout()

        fname = f"{output_folder}/hist_fdcall_{fd_call}_av_{available_set.replace(',', '_')}.png"
        plt.savefig(fname)
        plt.close()

def add_available_vehicles_column(df):
    """
    Adds an 'available_vehicles' column to the DataFrame for each call.
    For every row, a vehicle is available if its last CLEAR_TIME_fixed_character
    is before the current ALARM_fixed_character.
    """
    df = df.copy()
    df = df.sort_values("ALARM_fixed_character").reset_index(drop=True)
    vehicles = df["FD_response"].unique()

    # Track the last clear time for each vehicle (initialize with very early date)
    last_clear_time = {v: pd.Timestamp.min for v in vehicles}
    availability = []

    for idx, row in df.iterrows():
        current_time = row["ALARM_fixed_character"]
        available = [
            v for v in vehicles
            if last_clear_time[v] < current_time
        ]
        availability.append(sorted(available))
        last_clear_time[row["FD_response"]] = row["CLEAR_TIME_fixed_character"]

    df["available_vehicles"] = availability
    return df

def build_empirical_dispatch_distribution(df):
    """
    Builds an empirical distribution DataFrame for each unique combination of:
    (FD_call, tuple(sorted(available_vehicles)), FD_response).

    Returns:
        pd.DataFrame with columns:
        ['FD_call', 'available_set', 'FD_response', 'count', 'probability']
    """
    # Build records
    records = []
    for _, row in df.iterrows():
        if not row["available_vehicles"]:
            continue
        # ===== FIX: ensure the response vehicle is in the available set =====
        if row["FD_response"] not in row["available_vehicles"]:
            # Optional: print or log details for debug
            #print(f"Skip: FD_call={row['FD_call']} FD_response={row['FD_response']} not in available_vehicles={row['available_vehicles']} at {row['ALARM_fixed_character']}")
            continue
        # ===================================================================
        fd_call = row["FD_call"]
        fd_response = row["FD_response"]
        available = tuple(sorted(row["available_vehicles"]))
        records.append((fd_call, available, fd_response))

    # Count occurrences
    dispatch_counts = defaultdict(int)
    total_counts = defaultdict(int)
    for fd_call, available, fd_response in records:
        key = (fd_call, available, fd_response)
        dispatch_counts[key] += 1
        total_counts[(fd_call, available)] += 1

    # Build DataFrame
    summary_rows = []
    for (fd_call, available, fd_response), count in dispatch_counts.items():
        total = total_counts[(fd_call, available)]
        prob = count / total if total > 0 else 0
        summary_rows.append({
            "FD_call": fd_call,
            "available_set": ",".join(str(x) for x in available),
            "FD_response": fd_response,
            "count": count,
            "probability": prob
        })
    dispatch_df = pd.DataFrame(summary_rows)
    return dispatch_df

def select_vehicle_by_empirical_distribution(fd_call, available_vehicles, dispatch_df):
    """
    Selects a vehicle to dispatch based on empirical distribution, given the current call and available vehicles.

    Args:
        fd_call: The calling station.
        available_vehicles: List of currently available vehicles (sorted!).
        dispatch_df: DataFrame returned by build_empirical_dispatch_distribution.

    Returns:
        Chosen FD_response (vehicle/unit).
    """
    available_key = ",".join(sorted(available_vehicles))
    # Filter for relevant distribution
    subset = dispatch_df[
        (dispatch_df["FD_call"] == fd_call) &
        (dispatch_df["available_set"] == available_key)
    ]
    if subset.empty:
        raise ValueError(f"No empirical dispatch data for FD_call={fd_call} and available={available_key}")

    # Probabilities must sum to 1 (safe normalization)
    responses = subset["FD_response"].tolist()
    probabilities = subset["probability"].values
    probabilities = probabilities / probabilities.sum()

    selected_vehicle = np.random.choice(responses, p=probabilities)
    return selected_vehicle

def main():
    # === Load and process data ===
    input_file = "cleaned_Pumpers_C31_For_Analysis.xlsx"
    output_file = "cleaned_Pumpers_C31_With_A_Group.xlsx"

    # Load data
    df = pd.read_excel(input_file)
    # Make sure datetime columns are correct
    df["ALARM_fixed_character"] = pd.to_datetime(df["ALARM_fixed_character"])
    df["CLEAR_TIME_fixed_character"] = pd.to_datetime(df["CLEAR_TIME_fixed_character"])

    # Add available vehicles column
    df = add_available_vehicles_column(df)
    df.to_excel(output_file, index=False)
    print(f"Cleaned data saved to: {output_file}")

    # Build empirical dispatch distribution DataFrame
    dispatch_df = build_empirical_dispatch_distribution(df)
    dispatch_df.to_csv("empirical_dispatch_distribution.csv", index=False)
    print("Empirical dispatch distribution saved to: empirical_dispatch_distribution.csv")
    plot_dispatch_histograms(dispatch_df)
    print("Histograms for each dispatch distribution have been saved to the dispatch_histograms folder.")


if __name__ == "__main__":
    main()
