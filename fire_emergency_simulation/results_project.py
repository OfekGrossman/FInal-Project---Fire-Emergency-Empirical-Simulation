import numpy as np
import pandas as pd
from scipy import stats
import math
import pandas as pd
import glob
from typing import Dict, Tuple, List, Any
from sklearn import logger
from config import (NUM_PARAMETER_SETS, MAX_INTERARRIVAL_RANGE , MAX_TOTAL_SERVICE_RANGE , STEP_INNTERVAL, STEP_TOTAL_SERVICE, SERVICE_RT_RATIO)
from project import runProject 
from wining_scores import count_binary_score_for_set, save_summarize_results, get_CI_for_set
import csv
import os
from globals import globs



def get_final_results():
    results_dict = {}

    for interval_range in generate_interval_ranges():
        globs.interval_index = interval_range

        for total_service_range in generate_service_ranges():
            globs.total_services_index = total_service_range
            logger.info(f"Interval: {interval_range}, Service: {total_service_range}")

            #dict - polici1 vs ourpolicy2 : df of all the 5 set
            result_param = run_simulation_for_cell(interval_range, total_service_range)

            for comparison_name, df_p1_vs_p2 in result_param.items():
                print("Col name : ", df_p1_vs_p2.columns )
                print(df_p1_vs_p2)
                summary = summarize_cell_results(df_p1_vs_p2)
                save_summarize_results(summary, comparison_name, 'result_project.xlsx')
                #write_cell_to_csv(interval_range, total_service_range, comparison_name, summary)

    return results_dict

def generate_interval_ranges():
    for y in range(math.ceil((MAX_INTERARRIVAL_RANGE[1] - MAX_INTERARRIVAL_RANGE[0]) / STEP_INNTERVAL)):
        low = MAX_INTERARRIVAL_RANGE[0] + y * STEP_INNTERVAL
        yield (low, low + STEP_INNTERVAL)

def generate_service_ranges():
    for x in range(math.ceil((MAX_TOTAL_SERVICE_RANGE[1] - MAX_TOTAL_SERVICE_RANGE[0]) / STEP_TOTAL_SERVICE)):
        low = MAX_TOTAL_SERVICE_RANGE[0] + x * STEP_TOTAL_SERVICE
        yield (low, low + STEP_TOTAL_SERVICE)

def run_simulation_for_cell(interval_range, total_service_range):
    globs.set_index = 0
    service_range = tuple(s * SERVICE_RT_RATIO for s in total_service_range)
    response_range = tuple(s * (1 - SERVICE_RT_RATIO) for s in total_service_range)
    
    return runProject(interval_range, service_range, response_range)

def summarize_cell_results(result_df):
    
    return {
        'avg_queue_policy1': result_df['avg_policy1_queue'].mean(),
        'avg_queue_policy2': result_df['avg_policy2_queue'].mean(),
        'mean_improvement': result_df['mean_improvement'].mean(), 
        'win_percantage': result_df['win_percentage'].mean(),
        'win_score': count_binary_score_for_set(result_df, 'win_score'),
        'win_by_CI' : get_CI_for_set(result_df, 'CI_score_90','CI_teko_score_90' ),
        'win_by_Mean' : get_CI_for_set(result_df, 'CI_score_mean','CI_teko_score_mean'),
        'win_by_Test_statistique' : count_binary_score_for_set(result_df, 'Test_statistique_score')
    }


    

def save_cell_to_dict(results_dict, interval_range, service_range, summary):
    interval_key = f"{interval_range[0]}-{interval_range[1]}"
    service_key = f"{service_range[0]}-{service_range[1]}"
    results_dict.setdefault(interval_key, {})[service_key] = summary['avg_win_pct']

"""
def write_cell_to_csv(interval_range, service_range, summary):
    write_to_csv(
        interval_range[0], interval_range[1],
        service_range[0], service_range[1],
        (service_range[0] * SERVICE_RT_RATIO, service_range[1] * SERVICE_RT_RATIO),
        (service_range[0] * (1 - SERVICE_RT_RATIO), service_range[1] * (1 - SERVICE_RT_RATIO)),
        summary['avg_win_pct'], summary['mean_improvement'],
        summary['avg_queue_policy1'], summary['avg_queue_policy2']
    )
"""

def write_cell_to_csv(interval_range, service_range, comparison_name, summary):
    summary_dict = {
        'interval': [interval_range[0], interval_range[1]],
        'total_service': [service_range[0], service_range[1]],
        'service_range': (
            service_range[0] * SERVICE_RT_RATIO,
            service_range[1] * SERVICE_RT_RATIO
        ),
        'response_range': (
            service_range[0] * (1 - SERVICE_RT_RATIO),
            service_range[1] * (1 - SERVICE_RT_RATIO)
        ),
        'avg_win_pct': summary['avg_win_pct'],
        'mean_improvement': summary['mean_improvement'],
        'avg_queue_policy1': summary['avg_queue_policy1'],
        'avg_queue_policy2': summary['avg_queue_policy2'],
    }

    save_summarize_results(summary_dict, sheet_name=comparison_name, path='final_results_by_comparison.xlsx')




def write_to_csv(interval_low, interval_high, total_service_low, total_service_high, service_range, response_range, win_percentage, mean_improvement, avg_queue_policy1,avg_queue_policy2):
    csv_path="results.csv"
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write header if file does not exist
        if not file_exists:
            writer.writerow(["interval_low", "interval_high", "total_service_low", "total_service_high", "service_range", "response_range", "win_percentage", "mean_improvement","avg_queue_policy1" , "avg_queue_policy2"])
        writer.writerow([interval_low, interval_high, total_service_low, total_service_high, service_range, response_range, win_percentage, mean_improvement, avg_queue_policy1, avg_queue_policy2])

