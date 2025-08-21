import numpy as np
import pandas as pd
from scipy import stats
import time
import logging
from typing import Dict, Tuple, List, Any
from globals import globs
from models import Vehicle, PrecomputedTimes
from simulation import Simulation
from policies import DispatchPolicy
from config import (NUM_AREA, CV_SERVICE_RANGE, CV_RESPONSE_RANGE, EPSILON)
from empirical_dispatch import prepare_empirical_dispatch
from empirical_times import generate_empirical_samples_from_pmf, generate_interarrival_samples_by_fdcall

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("analysis")



def generate_random_times(num_of_events) -> PrecomputedTimes: #need change O.G
    """
    Generate  times from distribution.
    Args:
        mean_interarrival_times: Dictionary of interarrival times by area and vehicle
        mean_service_times: Dictionary of service times by area and vehicle
        mean_response_times: Dictionary of response times by area and vehicle
        num_events: Number of events to generate  
    Returns:
        PrecomputedTimes object containing all generated times
    """
    start_time = time.time()
    arrivals = {}
    services = {}
    responses = {}
   # times_dict = prepare_empirical_dispatch("empirical_dispatch_distribution.csv")
    times_dict = generate_empirical_samples_from_pmf("empirical_pmf.csv")
    inter_by_area = generate_interarrival_samples_by_fdcall("empirical_pmf_by_fd_call.csv")

    for area_id in range(NUM_AREA):
        # single arrival stream per area (use vehicle_id == 0 as the stream key)
        arrivals[(area_id, 0)] = inter_by_area["INTER_ARRIVAL_MIN"][area_id]

        for vehicle_id in range(NUM_AREA):

            # Generate service times
            services[(area_id, vehicle_id)] = times_dict["SERVICE_MIN"][(area_id, vehicle_id)] #list - generate num_of_events : service time for [area,vehicle]

            # Generate response times
            responses[(area_id, vehicle_id)] = times_dict["RESPONSE_TIME_MIN"][(area_id, vehicle_id)] #list - generate num_of_events : response time for [area,vehicle]

    return PrecomputedTimes(arrivals, services, responses)


def run_simulation_with_policies(vehicles: List[Vehicle], precomputed_times: PrecomputedTimes,
                               simulation_time: float, policies: List[DispatchPolicy]) -> List[Dict[str, Any]]:
    """
    Run simulations with multiple policies using the same random numbers.  
    Returns:
        List of dictionaries containing results for each policy
    """
    results = []
    global set_index, replication_index

    for policy in policies:
        policy_name = type(policy).__name__
        logger.info(f"Policy : {policy_name} ")
        logger.info(f"REP {globs.replication_index}")
        sim = Simulation(vehicles, policy, precomputed_times)
        sim.run(simulation_time)
        
        # Extract key metrics
        if sim.response_times:
            percentile_90 = np.percentile(sim.response_times, 90)
            mean_RT = np.mean(sim.response_times)
        else:
            percentile_90 = np.inf
            mean_RT = np.inf

        system_load = sim.total_service_time / (NUM_AREA * simulation_time)

        results.append({
            'policy': policy_name,
            'percentile_90': float(percentile_90),
            'mean_RT': float(mean_RT),
            'max_queue': int(sim.max_queue_size),
            'avg_queue': float(sim.avg_queue_size),  # NEW
            'system_load': float(system_load),
            'total_services': int(sim.total_services)
        })
        
    return results

