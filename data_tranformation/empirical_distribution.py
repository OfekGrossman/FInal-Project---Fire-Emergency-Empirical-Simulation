import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
import os

def plot_empirical_distributions(df,output_folder="ecdf_hist"):
    """
    Plots histograms and ECDFs for Response Time, Service Time, and Inter-Arrival Time
    for each FD_call / FD_response combination.
    """
    os.makedirs(output_folder, exist_ok=True)
    for (fd_call, fd_response), group in df.groupby(["FD_call", "FD_response"]):
        group = group.sort_values("ALARM_fixed_character").copy()
        group["Inter_Arrival_MIN"] = group["ALARM_fixed_character"].diff().dt.total_seconds() / 60

        rt = group["RESPONSE_TIME_MIN"].dropna()
        rt = rt[rt >= 0]
        st = group["SERVICE_MIN"].dropna()
        st = st[st >= 0]
        ia = group["Inter_Arrival_MIN"].dropna()
        ia = ia[ia >= 0]

        if len(rt) < 2 and len(st) < 2 and len(ia) < 2:
            continue

        fig, axs = plt.subplots(2, 3, figsize=(20, 8))
        fig.suptitle(f"FD_call: {fd_call} | FD_response: {fd_response}", fontsize=16)

        def plot_histogram(ax, data, title, color):
            if len(data) >= 2:
                counts, bins, _ = ax.hist(data, bins=20, color=color, edgecolor='black')
                ax.set_xticks(bins.round(1))
            ax.set_title(f"Histogram - {title}")
            ax.set_xlabel("Minutes")

        plot_histogram(axs[0, 0], rt, "Response Time", 'lightblue')
        axs[0, 0].set_ylabel("Frequency")
        plot_histogram(axs[0, 1], st, "Service Time", 'lightgreen')
        plot_histogram(axs[0, 2], ia, "Inter-Arrival Time", 'lightcoral')

        def plot_ecdf(ax, data, title):
            if len(data) >= 2:
                ecdf = ECDF(data)
                ax.plot(ecdf.x, ecdf.y, marker='.')
            ax.set_title(f"ECDF - {title}")
            ax.set_xlabel("Minutes")

        plot_ecdf(axs[1, 0], rt, "Response Time")
        axs[1, 0].set_ylabel("Probability")
        plot_ecdf(axs[1, 1], st, "Service Time")
        plot_ecdf(axs[1, 2], ia, "Inter-Arrival Time")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename = f"{output_folder}/ecdf_hist_{fd_call}_{fd_response}.png"
        plt.savefig(filename)
        plt.show()
        plt.close()

def compute_and_plot_interarrival_pmf(group, group_type, group_name, output_folder):
    import os
    from statsmodels.distributions.empirical_distribution import ECDF
    import matplotlib.pyplot as plt
    import pandas as pd

    group = group.sort_values("ALARM_fixed_character").copy()
    group["Inter_Arrival_MIN"] = group["ALARM_fixed_character"].diff().dt.total_seconds() / 60
    ia = group["Inter_Arrival_MIN"].dropna()
    ia = ia[ia >= 0]

    if len(ia) < 2:
        return pd.DataFrame()  # Skip small groups

    # === Compute PMF ===
    value_counts = ia.round(2).value_counts().sort_index()
    total = value_counts.sum()
    pmf_df = pd.DataFrame({
        group_type: group_name,
        "Value": value_counts.index,
        "Probability": value_counts.values / total
    })

    # === Plot ===
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Inter-Arrival PMF by {group_type}: {group_name}", fontsize=14)

    axs[0].bar(pmf_df["Value"], pmf_df["Probability"], color='lightcoral', edgecolor='black')
    axs[0].set_title("PMF Histogram")
    axs[0].set_xlabel("Minutes")
    axs[0].set_ylabel("Probability")

    ecdf = ECDF(ia)
    axs[1].plot(ecdf.x, ecdf.y, marker='.')
    axs[1].set_title("ECDF")
    axs[1].set_xlabel("Minutes")
    axs[1].set_ylabel("Probability")

    plt.tight_layout()
    plot_path = os.path.join(output_folder, f"interarrival_plot_{group_type}_{group_name}.png")
    plt.savefig(plot_path)
    plt.close()

    return pmf_df




def get_empirical_sampler(series):
    """
    Given a pandas Series of numeric values (e.g., response times),
    returns a function that samples new values based on the empirical distribution.
    """
    series = series.dropna()
    series = series[series >= 0]
    if len(series) < 2:
        return None
    ecdf = ECDF(series.sort_values())
    inv_cdf = interp1d(
        ecdf.y,
        ecdf.x,
        bounds_error=False,
        fill_value=(ecdf.x[0], ecdf.x[-1])
    )
    return lambda n=1: inv_cdf(np.random.uniform(0, 1, n))

def save_empirical_parameters_wide(df, output_file="empirical_parameters_wide.csv"):
    """
    For each (FD_call, FD_response) combination, saves a single row containing:
      - FD_call
      - FD_response
      - RESPONSE_TIME_MIN (list as string)
      - SERVICE_MIN (list as string)
      - INTER_ARRIVAL_MIN (list as string)
    """
    records = []
    for (fd_call, fd_response), group in df.groupby(["FD_call", "FD_response"]):
        group = group.sort_values("ALARM_fixed_character").copy()
        group["Inter_Arrival_MIN"] = group["ALARM_fixed_character"].diff().dt.total_seconds() / 60

        rt_list = [round(float(x), 2) for x in group["RESPONSE_TIME_MIN"].dropna() if x >= 0]
        st_list = [round(float(x), 2) for x in group["SERVICE_MIN"].dropna() if x >= 0]
        ia_list = [round(float(x), 2) for x in group["Inter_Arrival_MIN"].dropna() if x >= 0]

        records.append({
            "FD_call": fd_call,
            "FD_response": fd_response,
            "RESPONSE_TIME_MIN": str(rt_list),
            "SERVICE_MIN": str(st_list),
            "INTER_ARRIVAL_MIN": str(ia_list)
        })
    param_df = pd.DataFrame(records)
    param_df.to_csv(output_file, index=False)
    print(f"Empirical parameters (wide format, 2 decimals) saved to: {output_file}")

def save_empirical_pmf(df, output_file="empirical_pmf.csv"):
    """
    For each (FD_call, FD_response) combination and each parameter,
    computes the empirical PMF (value and probability), and saves to CSV.
    """
    records = []
    for (fd_call, fd_response), group in df.groupby(["FD_call", "FD_response"]):
        group = group.sort_values("ALARM_fixed_character").copy()
        group["Inter_Arrival_MIN"] = group["ALARM_fixed_character"].diff().dt.total_seconds() / 60

        param_map = {
            "RESPONSE_TIME_MIN": group["RESPONSE_TIME_MIN"].dropna(),
            "SERVICE_MIN": group["SERVICE_MIN"].dropna(),
            "INTER_ARRIVAL_MIN": group["Inter_Arrival_MIN"].dropna()
        }
        for param, values in param_map.items():
            values = values[values >= 0].round(2)  # <--- changed from .round(3) to .round(2)
            value_counts = values.value_counts().sort_index()
            total = value_counts.sum()
            for val, count in value_counts.items():
                records.append({
                    "FD_call": fd_call,
                    "FD_response": fd_response,
                    "Parameter": param,
                    "Value": val,
                    "Probability": count / total
                })
    pmf_df = pd.DataFrame(records)
    pmf_df.to_csv(output_file, index=False)
    print(f"Empirical PMFs (2 decimals) saved to: {output_file}")

def save_empirical_pmf_by_fd_call(df, output_file="empirical_pmf_by_fd_call.csv"):
    """
    For each fd_call and each parameter,
    computes the empirical PMF (value and probability), and saves to CSV.
    """
    records = []
    for fd_call, group in df.groupby(["FD_call"]):
        group = group.sort_values("ALARM_fixed_character").copy()
        group["Inter_Arrival_MIN"] = group["ALARM_fixed_character"].diff().dt.total_seconds() / 60

        param_map = {
            "INTER_ARRIVAL_MIN": group["Inter_Arrival_MIN"].dropna()
        }
        for param, values in param_map.items():
            values = values[values >= 0].round(2)  # <--- changed from .round(3) to .round(2)
            value_counts = values.value_counts().sort_index()
            total = value_counts.sum()
            for val, count in value_counts.items():
                records.append({
                    "FD_call": fd_call[0],
                    "Parameter": param,
                    "Value": val,
                    "Probability": count / total
                })
    pmf_df = pd.DataFrame(records)
    pmf_df.to_csv(output_file, index=False)
    print(f"Empirical PMFs by fd_call(2 decimals) saved to: {output_file}")


def generate_interarrival_by_fdcall(df, output_folder="interarrival_by_fdcall", output_csv="interarrival_fdcall_pmf.csv"):
    os.makedirs(output_folder, exist_ok=True)
    df["ALARM_fixed_character"] = pd.to_datetime(df["ALARM_fixed_character"])
    df = df.sort_values("ALARM_fixed_character").copy()

    all_records = []
    for fd_call, group in df.groupby("FD_call"):
        pmf_df = compute_and_plot_interarrival_pmf(group, "FD_call", str(fd_call), output_folder)
        if not pmf_df.empty:
            all_records.append(pmf_df)

    if all_records:
        final_df = pd.concat(all_records, ignore_index=True)
        final_df.to_csv(os.path.join(output_folder, output_csv), index=False)
        print(f"Combined inter-arrival PMF by FD_call saved to: {output_folder}/{output_csv}")
    else:
        print("No valid FD_call groups found.")



def generate_interarrival_by_fdresponse(df, output_folder="interarrival_by_fdresponse", output_csv="interarrival_fdresponse_pmf.csv"):
    os.makedirs(output_folder, exist_ok=True)
    df["ALARM_fixed_character"] = pd.to_datetime(df["ALARM_fixed_character"])
    df = df.sort_values("ALARM_fixed_character").copy()

    all_records = []
    for fd_response, group in df.groupby("FD_response"):
        pmf_df = compute_and_plot_interarrival_pmf(group, "FD_response", str(fd_response), output_folder)
        if not pmf_df.empty:
            all_records.append(pmf_df)

    if all_records:
        final_df = pd.concat(all_records, ignore_index=True)
        final_df.to_csv(os.path.join(output_folder, output_csv), index=False)
        print(f"Combined inter-arrival PMF by FD_response saved to: {output_folder}/{output_csv}")
    else:
        print("No valid FD_response groups found.")




def main():
    # === File paths ===
    input_file = "Pumpers_C31_For_Analysis.xlsx"
    cleaned_file = "cleaned_Pumpers_C31_For_Analysis.xlsx"
    summary_file = "fd_combination_summary.csv"

    # === Load and prepare data ===
    df = pd.read_excel(input_file)
    df["ALARM_fixed_character"] = pd.to_datetime(df["ALARM_fixed_character"])
    df = df.sort_values("ALARM_fixed_character")

    # === Remove invalid rows ===
    initial_rows = len(df)
    neg_response = (df["RESPONSE_TIME_MIN"] < 0).sum()
    neg_service = (df["SERVICE_MIN"] < 0).sum()

    df = df[(df["RESPONSE_TIME_MIN"] >= 0) & (df["SERVICE_MIN"] >= 0)]
    after_rows = len(df)

    print(f"Initial rows: {initial_rows}")
    print(f"Removed rows with negative RESPONSE_TIME_MIN: {neg_response}")
    print(f"Removed rows with negative SERVICE_MIN: {neg_service}")
    print(f"Remaining rows after cleanup: {after_rows}")

    # === Save cleaned data ===
    df.to_excel(cleaned_file, index=False)
    print(f"Cleaned data saved to: {cleaned_file}")

    # === Save empirical parameters in wide format for simulation ===
    save_empirical_parameters_wide(df, output_file="empirical_parameters_wide.csv")

    # === Save empirical PMFs for simulation ===
    save_empirical_pmf(df, output_file="../fire_emergency_simulation/empirical_pmf.csv")

    # === Save empirical PMFs by fd_call for simulation ===
    save_empirical_pmf_by_fd_call(df, output_file="../fire_emergency_simulation/empirical_pmf_by_fd_call.csv")

    # === Create samplers and summary ===
    sampler_dict = {}
    summary_records = []

    for (fd_call, fd_response), group in df.groupby(["FD_call", "FD_response"]):
        group = group.sort_values("ALARM_fixed_character").copy()
        group["Inter_Arrival_MIN"] = group["ALARM_fixed_character"].diff().dt.total_seconds() / 60

        rt_valid = group["RESPONSE_TIME_MIN"].dropna()
        st_valid = group["SERVICE_MIN"].dropna()
        ia_valid = group["Inter_Arrival_MIN"].dropna()

        sampler_dict[(fd_call, fd_response)] = {
            "response_time_sampler": get_empirical_sampler(rt_valid),
            "service_time_sampler": get_empirical_sampler(st_valid),
            "interarrival_time_sampler": get_empirical_sampler(ia_valid)
        }

        summary_records.append({
            "FD_call": fd_call,
            "FD_response": fd_response,
            "ResponseTime_Count": (rt_valid >= 0).sum(),
            "ServiceTime_Count": (st_valid >= 0).sum(),
            "InterArrival_Count": (ia_valid >= 0).sum()
        })

    # === Save summary ===
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary CSV saved to: {summary_file}")

    # === Generate visualizations ===
    """
    print("Generating ECDF & histogram plots...")
    plot_empirical_distributions(df)

    # === Generate inter-arrival plots independently by FD_call and FD_response ===
    generate_interarrival_by_fdcall(df)
    generate_interarrival_by_fdresponse(df)
    """
if __name__ == "__main__":
    main()
