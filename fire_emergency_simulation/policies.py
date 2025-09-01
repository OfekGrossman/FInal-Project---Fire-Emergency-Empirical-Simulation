from abc import ABC, abstractmethod
from typing import Set, Dict, Optional, List, Tuple
import random
from scipy import stats
import numpy as np
from models import Vehicle, Event, EventLog, PrecomputedTimes, EventType
from empirical_dispatch import prepare_empirical_dispatch, empirical_dispatch




class DispatchPolicy(ABC):
    """
    Abstract base class for vehicle dispatch policies.
    All concrete policies must implement the select_vehicle method.
    """

    @abstractmethod
    def compute_parameters(self, precomputed_times: PrecomputedTimes, vehicles: List[Vehicle], 
                           num_area: int, mean_response_times: Dict[int, Dict[int, float]]) -> None:
        pass
    
    @abstractmethod
    def select_vehicle(self, area_id: int, available_vehicles: Set[int]) -> Optional[int]:
        """
        Select a vehicle to dispatch to the given area.
        
        Args:
            area_id: ID of the area requiring service
            available_vehicles: Set of available vehicle IDs
            
        Returns:
            ID of the selected vehicle, or None if no vehicle is available
        """
        pass

class EmpiricalDispatch(DispatchPolicy):

    def compute_parameters(self, precomputed_times: PrecomputedTimes, vehicles: List[Vehicle], num_area: int,
                           mean_response_times: Dict[int, Dict[int, float]]) -> None:
        self.dispatch_dict = prepare_empirical_dispatch("empirical_dispatch_distribution.csv")

    def select_vehicle(self, area_id: int, available_vehicles: Set[int]) -> Optional[int]:
   
        if not available_vehicles:
            return None
        return empirical_dispatch(self.dispatch_dict, area_id, list(available_vehicles))
  

class LBR(DispatchPolicy):
    """
    Policy that selects vehicle based on prioritization parameters.
    prioritization_parameters: Matrix of prioritization parameters
    """
    prioritization_parameters: Dict[int, Dict[int, float]]

    def compute_parameters(
            self,
            precomputed_times: PrecomputedTimes,
            vehicles: List[Vehicle],
            num_area: int,
            mean_response_times: Dict[int, Dict[int, float]]
    ) -> None:
        self.prioritization_parameters = {}
        for area_id in range(num_area):
            self.prioritization_parameters[area_id] = {}
            for vehicle_id, vehicle in enumerate(vehicles):
                # Service/response stay per (area, vehicle)
                services = precomputed_times.services[(area_id, vehicle_id)]
                responses = precomputed_times.responses[(area_id, vehicle_id)]

                # Make sure both lists have the same length
                min_length = min(len(services), len(responses))
                services = services[:min_length]
                responses = responses[:min_length]

                # Total service time (service + response)
                total_times = [services[i] + responses[i] for i in range(min_length)]

                # NEW: λ from the single per-area interarrival stream (area_id, 0)
                iarea = precomputed_times.arrivals.get((area_id, 0), [])
                if iarea and sum(iarea) > 0:
                    # MLE for Poisson rate = n / sum(inter-arrivals) == 1 / mean
                    lambda_j = len(iarea) / sum(iarea)
                else:
                    lambda_j = 0.0001  # small epsilon to avoid div-by-zero / empty lists

                # === same formula as before ===
                exp_term = np.mean([np.exp(-lambda_j * total_time) for total_time in total_times])
                percentile_95 = np.percentile(responses, 95)
                resp_term = 1 / percentile_95

                self.prioritization_parameters[area_id][vehicle_id] = exp_term * resp_term
    
    def select_vehicle(self, area_id: int, available_vehicles: Set[int]) -> Optional[int]:
        """
        Select vehicle with the highest prioritization parameter.
        
        Args:
            area_id: ID of the area requiring service
            available_vehicles: Set of available vehicle IDs
            
        Returns:
            ID of the selected vehicle, or None if no vehicle is available
        """
        if not available_vehicles:
            return None
            
        return max(available_vehicles,
                  key=lambda vid: self.prioritization_parameters[area_id][vid])






class MinP95(DispatchPolicy):
    """
    Policy that selects the available vehicle with the minimum
    95th-percentile response time for the requested area.
    """
    p95_response_by_area_vehicle: Dict[int, Dict[int, float]]

    def compute_parameters(
        self,
        precomputed_times: PrecomputedTimes,
        vehicles: List[Vehicle],
        num_area: int,
        mean_response_times: Dict[int, Dict[int, float]] = None,  # unused; kept for interface parity
    ) -> None:
        self.p95_response_by_area_vehicle = {}
        for area_id in range(num_area):
            self.p95_response_by_area_vehicle[area_id] = {}
            for vehicle_id, _ in enumerate(vehicles):
                responses = precomputed_times.responses.get((area_id, vehicle_id), [])
                if responses:
                    p95 = float(np.percentile(responses, 95))
                else:
                    p95 = float("inf")  # no data → never prefer this vehicle
                self.p95_response_by_area_vehicle[area_id][vehicle_id] = p95

    def select_vehicle(self, area_id: int, available_vehicles: Set[int]) -> Optional[int]:
        if not available_vehicles:
            return None
        # Choose argmin by precomputed P95; tie-break by vehicle id (stable & deterministic)
        return min(
            available_vehicles,
            key=lambda vid: (self.p95_response_by_area_vehicle[area_id][vid], vid)
        )

