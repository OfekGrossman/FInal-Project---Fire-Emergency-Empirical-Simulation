from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
from enum import Enum


class EventType(Enum):
    """Types of events in the simulation system."""
    ARRIVAL = "arrival"
    COMPLETION = "completion"
    QUEUE_PROCESSING = "queue_processing"


@dataclass
class Vehicle:
    """
    Represents an emergency vehicle that responds to events.
    
    Attributes:
        id: Unique identifier for the vehicle
    """
    id: int
    total_rate: int = 0
    
    #need change O.G
    def generate_service_time(self, area_id: int) -> Tuple[float, float]:
        """
        Generate service and response times for a call from the given area.
        
        Args:
            area_id: ID of the area generating the call
            
        Returns:
            Tuple of (service_time, response_time)
        """
        service_time = 0
        response_time = 0
        return service_time, response_time

    @property
    def total_arrival_rate(self) -> float:
        """Total arrival rate across all areas."""
        return sum(self.arrival_rates.values())
    
    def calculate_total_rate(self, arrival_time ):
        pass


class Event:
    """
    Represents a discrete event in the simulation.
    
    Attributes:
        time: Time when the event occurs
        event_type: Type of event (arrival, completion, etc.)
        area_id: ID of the area associated with this event
        vehicle_id: ID of the vehicle associated with this event
        arrival_time: Original arrival time for tracking waiting time
    """
    def __init__(self, time: float, event_type: EventType, 
                 area_id: Optional[int] = None,
                 vehicle_id: Optional[int] = None, 
                 arrival_time: Optional[float] = None):
        self.time = time
        self.event_type = event_type
        self.area_id = area_id
        self.vehicle_id = vehicle_id
        self.arrival_time = arrival_time

    def __lt__(self, other):
        """For priority queue ordering."""
        return self.time < other.time

    def __eq__(self, other):
        """Define equality for comparison."""
        return self.time == other.time


@dataclass
class EventLog:
    """
    Stores information about a logged event.
    
    Attributes:
        time_of_event: When the event occurred
        time_of_service: Service time for the event
        response_time: Response time for the event
        available_engines: List of available vehicles at event time
        chosen_engine: Which vehicle was selected
        mean_response_times: Current response time estimates
        prioritization_parameters: Current prioritization parameters
    """
    time_of_event: float
    time_of_service: float
    response_time: float
    available_engines: List[int]
    chosen_engine: int
    mean_response_times: Dict[int, float]
    prioritization_parameters: Dict[int, float]


@dataclass
class PrecomputedTimes:
    """
    Stores precomputed random times for simulation.
    
    Attributes:
        arrivals: Dictionary of arrival times by (area_id, vehicle_id)
        services: Dictionary of service times by (area_id, vehicle_id)
        responses: Dictionary of response times by (area_id, vehicle_id)
    """
    arrivals: Dict[Tuple[int, int], List[float]]
    services: Dict[Tuple[int, int], List[float]]
    responses: Dict[Tuple[int, int], List[float]]