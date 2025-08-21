import pandas as pd
import numpy as np
import os
from scipy import stats
from globals import globs
from scipy.stats import wilcoxon
import openpyxl
from config import NUM_REPLICATIONS, NUM_PARAMETER_SETS

import openpyxl

def save_summarize_results(summary_dict, sheet_name, path):
    results_df = pd.DataFrame([summary_dict])
    results_df.insert(0, '(Param, SET)_index', 
                  '({0}, {1})'.format(globs.set_index, globs.replication_index))
    results_df.insert(0, '(Y interval, X service)', 
                  '({0}, {1})'.format(globs.interval_index, globs.total_services_index))

    file_path = os.path.join(globs.folder_path, path)
    if os.path.exists(file_path): 
        with pd.ExcelWriter(file_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            try:
                existing_df = pd.read_excel(file_path, sheet_name=sheet_name)
                combined_df = pd.concat([existing_df, results_df], ignore_index=True)
            except ValueError: # creating new Sheet if needed
                combined_df = results_df

            combined_df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        # אם הקובץ לא קיים, צור Sheet ראשון
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name=sheet_name, index=False)



# Win Score and Test Score
def count_binary_score_for_set(result_df, col_name):
    return (result_df[col_name].sum() / NUM_PARAMETER_SETS)*100

# Win score percentage
def get_win_score_percentage(rel_improvements) : 
    count_win = sum(i > 0 for i in rel_improvements) 
    percentage_win = (count_win / NUM_REPLICATIONS )* 100
    if count_win > (NUM_REPLICATIONS/2) :
        return 1, percentage_win
    return 0, percentage_win


# CI score
def get_score_and_save_CI(list_p1 : list, list_p2 : list, name_p1_vs_p2, path):
    p2_wins = 0
    count_teko = 0 
    p1_low, p1_high = get_CI(list_p1)
    p2_low, p2_high = get_CI(list_p2)

    if p2_high < p1_low :
        p2_wins += 1
    elif p1_high < p2_low :
        pass
    else :
        count_teko += 1

    summary_dict = {
    'CI_p1': f"[{p1_low:.2f} , {p1_high:.2f}]",
    'CI_p2': f"[{p2_low:.2f} , {p2_high:.2f}]",
    'p2_win': p2_wins,
    'teko': count_teko
    }
    save_summarize_results(summary_dict, name_p1_vs_p2, path)

    return p2_wins, count_teko

def get_CI_for_set(result_df, str_score, str_tekko):
    count_tekko = result_df[ str_tekko].sum()
    count_win = result_df[str_score].sum()
    count_Signification_statistique = NUM_PARAMETER_SETS - count_tekko
    
    if count_win > int(count_Signification_statistique/ 2)  :
      return 1
    elif count_tekko > int(count_Signification_statistique/ 2) :
        return 0
    return -1

def get_CI(data: list[float], confidence: float = 0.90) -> tuple[float, float]:
    if len(data) == 0:
        return (np.nan, np.nan)
    mean = np.mean(data)
    sem = stats.sem(data)  # סטיית תקן של הממוצע
    margin = stats.t.ppf((1 + confidence) / 2, len(data) - 1) * sem
    return (mean - margin, mean + margin)


#Test Statistic Score
def get_statistique_score(policy1_percentiles, policy2_percentiles, name_p1_vs_p2, path):
    result = wilcoxon_one_sided(policy1_percentiles, policy2_percentiles, 0.10)
    save_summarize_results(result, name_p1_vs_p2, path)
    if result["reject_null"] : #win
        return 1
    return 0


def wilcoxon_one_sided(policy1_values, policy2_values, alpha):
    if len(policy1_values) != len(policy2_values):
        raise ValueError("Lists are not in same lenght")

    differences = np.array(policy1_values) - np.array(policy2_values)

    if np.all(differences == 0):
        return {
            "mean_difference": 0.0,
            "statistic": None,
            "p_value": 1.0,
            "reject_null": False
        }
    stat, p_value = wilcoxon(differences, alternative='greater')

    return {
        "mean_difference": np.mean(differences),
        "statistic": stat,
        "p_value": p_value,
        "reject_null": p_value < alpha
    }


