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


class MeanRT(DispatchPolicy):
    """
    Policy that selects vehicle with the lowest mean response time.
    
    mean_response_times: Matrix of mean response times by area and vehicle
    """
    mean_response_times: Dict[int, Dict[int, float]]

    def compute_parameters(self, precomputed_times: PrecomputedTimes, vehicles: List[Vehicle], 
                           num_area: int, mean_response_times: Dict[int, Dict[int, float]]) -> None:
        self.mean_response_times = mean_response_times
    
    def select_vehicle(self, area_id: int, available_vehicles: Set[int]) -> Optional[int]:
        """
        Select vehicle with lowest mean response time.
        Args:
            area_id: ID of the area requiring service
            available_vehicles: Set of available vehicle IDs
        Returns:
            ID of the selected vehicle, or None if no vehicle is available
        """
        if not available_vehicles:
            return None
            
        return min(available_vehicles,
                  key=lambda vid: self.mean_response_times[area_id][vid])

class Percentil_95(DispatchPolicy):
    """
    Policy that selects vehicle with the lowest percentile 95 response time.
    """
    percentile_95: Dict[int, Dict[int, float]]


    def compute_parameters(self, precomputed_times: PrecomputedTimes, vehicles: List[Vehicle], num_area: int,
                        mean_response_times: Dict[int, Dict[int, float]]) -> None:
        
        self.percentile_95 = {}
        
        for area_id in range(num_area):
            self.percentile_95[area_id] = {}  
            for vehicle_id, vehicle in enumerate(vehicles):

                # Calculate 95th percentile from lognormal distribution
                mean_response = mean_response_times[area_id][vehicle_id]
                cv_response = vehicle.response_cvs[area_id]

                # Convert to lognormal parameters
                sigma = np.sqrt(np.log(1 + cv_response ** 2))
                mu = np.log(mean_response) - 0.5 * sigma ** 2

                # Use SciPy's lognormal PPF function
                percentile_95 = stats.lognorm.ppf(0.95, s=sigma, scale=np.exp(mu))

                self.percentile_95[area_id][vehicle_id] = percentile_95
    
    
    def select_vehicle(self, area_id: int, available_vehicles: Set[int]) -> Optional[int]:
        """
        Select vehicle with lowest percentile response time.
        Args:
            area_id: ID of the area requiring service
            available_vehicles: Set of available vehicle IDs
        Returns:
            ID of the selected vehicle, or None if no vehicle is available
        """
        if not available_vehicles:
            return None
            
        return min(available_vehicles,
                  key=lambda vid: self.percentile_95[area_id][vid])



