from itertools import combinations
import logging
import pandas as pd
import numpy as np
import os
from typing import Set, Dict, Optional, List, Tuple,Any
from analysis import run_simulation_with_policies, generate_random_times
from policies import LBR, EmpiricalDispatch, MinP95
from models import Vehicle
from config import NUM_AREA, NUM_PARAMETER_SETS, NUM_REPLICATIONS, SIMULATION_TIME, NUM_SAMPLES
from globals import globs
from plots import generate_policy_comparison_plots
from wining_scores import get_statistique_score, get_score_and_save_CI


# ---------------------------------------------------
# Logging
# ---------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("runProject")


# ---------------------------------------------------
# Append one row per replication to a single Excel
# ---------------------------------------------------
def append_replication_row(row: dict, path: str = "replication_results.xlsx", sheet: str = "results") -> None:
    """Append a single dict row to one Excel sheet (create file/sheet on first write)."""
    df = pd.DataFrame([row])
    if not os.path.exists(path):
        df.to_excel(path, index=False, sheet_name=sheet)
        return

    # append without duplicating headers
    with pd.ExcelWriter(path, mode="a", engine="openpyxl", if_sheet_exists="overlay") as writer:
        book = writer.book
        if sheet in book.sheetnames:
            startrow = book[sheet].max_row
        else:
            startrow = 0
        df.to_excel(writer, index=False, header=(startrow == 0), sheet_name=sheet, startrow=startrow)


# ---------------------------------------------------
# Vehicles
# ---------------------------------------------------
def generate_vehicles(precomputed):
    """
    Build Vehicle objects and compute a sensible total_rate per vehicle:
    total_rate = 1 / mean(service + response) aggregated across areas for this vehicle.
    This avoids touching models.py even if Vehicle(total_rate) is required.
    """
    EPS = 1e-4
    vehicles = []

    # infer counts from precomputed (more robust than assuming NUM_AREA)
    num_vehicles = max(v for (_, v) in precomputed.services.keys()) + 1
    num_areas = max(a for (a, _) in precomputed.services.keys()) + 1

    for vid in range(num_vehicles):
        totals = []
        for area_id in range(num_areas):
            s = precomputed.services[(area_id, vid)]
            r = precomputed.responses[(area_id, vid)]
            n = min(len(s), len(r))
            if n > 0:
                totals.extend([s[i] + r[i] for i in range(n)])

        if totals:
            mean_total = float(np.mean(totals))
            total_rate = (1.0 / mean_total) if mean_total > 0 else EPS
        else:
            total_rate = EPS

        vehicles.append(Vehicle(vid, total_rate))
    return vehicles


# ---------------------------------------------------
# Replications runner
# ---------------------------------------------------
def run_replications():
    globs.replication_index = 0
    policies = [EmpiricalDispatch(), MinP95(), LBR()]
    all_policies = policies
    policy_rep_results = {type(p).__name__: [] for p in all_policies}

    for rep in range(NUM_REPLICATIONS):
        globs.replication_index += 1
        precomputed = generate_random_times(NUM_SAMPLES)
        vehicles = generate_vehicles(precomputed)
        results = run_simulation_with_policies(vehicles, precomputed, SIMULATION_TIME, all_policies)
        append_three_policy_row(results, rep)
        for policy, result in zip(all_policies, results):
            policy_rep_results[type(policy).__name__].append(result)

    summarized_results = {}
    for p1, p2 in combinations(all_policies, 2):
        p1_name = type(p1).__name__
        p2_name = type(p2).__name__
        p1_results = policy_rep_results[p1_name]
        p2_results = policy_rep_results[p2_name]
        name = f"{p1_name} vs {p2_name}"
        summarized_results[name] = summarize_replication_results(p1_results, p2_results, name)

    return summarized_results


# ---------------------------------------------------
# Summarize per-rep and write one row per replication
# ---------------------------------------------------
def summarize_replication_results(p1_results, p2_results, name_p1_vs_p2):
    """
    Build one Excel row per replication with rich metrics:
    P90, mean RT, max queue, avg queue (if provided by analysis/simulation),
    system load, total services, and diffs (P2 - P1). Also include a simple winner flag.
    """
    policy1_percentiles, policy2_percentiles = [], []
    policy1_mean_RT, policy2_mean_RT = [], []
    policy1_queues, policy2_queues = [], []
    rel_improvements = []
    policy1_loads, policy2_loads = [], []

    for rep_idx, (p1_result, p2_result) in enumerate(zip(p1_results, p2_results), start=1):
        percentile_p1, percentile_p2 = p1_result['percentile_90'], p2_result['percentile_90']
        mean1, mean2 = p1_result['mean_RT'], p2_result['mean_RT']

        policy1_percentiles.append(float(percentile_p1))
        policy2_percentiles.append(float(percentile_p2))
        policy1_mean_RT.append(mean1)
        policy2_mean_RT.append(mean2)

        p1_max_q = int(p1_result.get('max_queue', 0))
        p2_max_q = int(p2_result.get('max_queue', 0))
        p1_avg_q = float(p1_result.get('avg_queue', float('nan')))
        p2_avg_q = float(p2_result.get('avg_queue', float('nan')))
        p1_load  = float(p1_result.get('system_load', float('nan')))
        p2_load  = float(p2_result.get('system_load', float('nan')))
        p1_serv  = int(p1_result.get('total_services', 0))
        p2_serv  = int(p2_result.get('total_services', 0))

        # single Excel row for this replication
        row = {
            "comparison": name_p1_vs_p2,
            "param_set": globs.set_index,          # helpful context if you iterate parameter sets
            "replication": rep_idx,

            "p1_percentile_90": float(percentile_p1),
            "p2_percentile_90": float(percentile_p2),
            "diff_percentile_90": float(percentile_p2 - percentile_p1),

            "p1_mean_RT": float(mean1),
            "p2_mean_RT": float(mean2),
            "diff_mean_RT": float(mean2 - mean1),

            "p1_max_queue": p1_max_q,
            "p2_max_queue": p2_max_q,
            "diff_max_queue": p2_max_q - p1_max_q,

            "p1_avg_queue": p1_avg_q,
            "p2_avg_queue": p2_avg_q,
            "diff_avg_queue": p2_avg_q - p1_avg_q,

            "p1_system_load": p1_load,
            "p2_system_load": p2_load,
            "diff_system_load": p2_load - p1_load,

            "p1_total_services": p1_serv,
            "p2_total_services": p2_serv,
            "diff_total_services": p2_serv - p1_serv,

            "win_p90": 1 if percentile_p2 < percentile_p1 else 0,
            "win_mean": 1 if mean2 < mean1 else 0,
        }
        append_replication_row(row, path="replication_results.xlsx", sheet="results")

        # keep a few aggregates in-memory (if you still consume summarized_results elsewhere)
        rel_improvements.append((percentile_p1 - percentile_p2) / max(percentile_p1, 1e-9) * 100)
        policy1_queues.append(p1_max_q)
        policy2_queues.append(p2_max_q)
        policy1_loads.append(p1_load)
        policy2_loads.append(p2_load)

    # One-sided Wilcoxon at alpha=0.10: tests whether Policy 2 < Policy 1 (better if lower P90)
    _wilcoxon_win = get_statistique_score(
        policy1_percentiles, policy2_percentiles,
        name_p1_vs_p2, path="replication_results.xlsx"
    )

    # 90% CI overlap summary (p2_win/teko saved). Sheet name = comparison name.
    _p2_wins, _tekos = get_score_and_save_CI(
        policy1_percentiles, policy2_percentiles,
        name_p1_vs_p2, path="replication_results.xlsx"
    )
    # light in-memory summary (no extra Excel files produced)

    return {
        'avg_policy1_load': float(np.mean(policy1_loads)) if policy1_loads else np.nan,
        'avg_policy2_load': float(np.mean(policy2_loads)) if policy2_loads else np.nan,
        'avg_policy1_queue': float(np.mean(policy1_queues)) if policy1_queues else np.nan,
        'avg_policy2_queue': float(np.mean(policy2_queues)) if policy2_queues else np.nan,
        'mean_improvement': float(np.mean(rel_improvements)) if rel_improvements else np.nan,
    }
def append_three_policy_row(results: List[Dict[str, Any]], rep_idx: int):
    """
    Writes one row per replication with all three policies' results side-by-side.
    Assumes order: [Empirical, MinP95, LBR]
    """
    result_dict = {
        "replication": rep_idx,
    }
    for r in results:
        name = r['policy']
        result_dict.update({
            f"{name}_p90": r['percentile_90'],
            f"{name}_mean_RT": r['mean_RT'],
            f"{name}_avg_queue": r['avg_queue'],
            f"{name}_max_queue": r['max_queue'],
            f"{name}_system_load": r['system_load'],
            f"{name}_total_services": r['total_services']
        })

    append_replication_row(result_dict, path="replication_results.xlsx", sheet="three_policy_results")

# ---------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------
def runProject():
    logger.info("Starting full policy comparison simulation...")
    # NEW: start fresh results file every run
    RESULTS_XLSX = "replication_results.xlsx"
    try:
        if os.path.exists(RESULTS_XLSX):
            os.remove(RESULTS_XLSX)
            logger.info(f"Deleted previous results file: {RESULTS_XLSX}")
    except Exception as e:
        logger.warning(f"Could not delete {RESULTS_XLSX} (is it open?): {e}")
    results_parm = []
    for param_set in range(NUM_PARAMETER_SETS):
        globs.set_index = param_set + 1  # keep context in the output rows
        result = run_replications()
        results_parm.append(result)

    # If you still need a returned object for further Python processing:
    final = {}
    for result_param in results_parm:
        for comparison_name, set_result in result_param.items():
            final.setdefault(comparison_name, [])
            final[comparison_name].append(set_result)
    final_dfs = {comparison_name: pd.DataFrame(list_set) for comparison_name, list_set in final.items()}
    generate_policy_comparison_plots(
        xlsx_path=RESULTS_XLSX,
        sheet="results",
        output_dir="plots"
    )

    return final_dfs



