import heapq
import time
import warnings
import pandas as pd
import numpy as np
from typing import List, Dict, Set, Optional, Tuple, Any
import logging
from models import Vehicle, Event, EventLog, PrecomputedTimes, EventType
from policies import DispatchPolicy
from config import NUM_AREA, NUM_SAMPLES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("simulation")


class Simulation:
    """
    Core simulation engine for emergency dispatch system.
    
    This class handles the discrete event simulation including event scheduling,
    vehicle dispatching, and statistics collection.
    """
    
    def __init__(self, vehicles: List[Vehicle], dispatch_policy: DispatchPolicy,
                 precomputed_times: PrecomputedTimes):
        """
        Initialize simulation with entities and configuration.
        
        Args:
            vehicles: List of Vehicle objects representing available units
            dispatch_policy: Policy to use for vehicle selection
            precomputed_times: Pre-generated random times for reproducibility
        """
        self.total_service_time = 0
        self.vehicles = vehicles
        self.dispatch_policy = dispatch_policy
        self.precomputed_times = precomputed_times
        self.available_vehicles = set(range(len(vehicles)))
        self.events = []
        self.current_time = 0
        self.response_times = []
        self.waiting_queue = []
        self.max_queue_size = 0
        self._queue_area = 0.0  # time-integral of queue length
        self._last_time_for_avg = 0.0  # last timestamp we accounted queue length for
        self.avg_queue_size = 0.0  # will be filled at the end of run()
        self.total_services = 0

        # Track usage of precomputed times
        self.time_indices = {(i, j): 0 for i in range(NUM_AREA) for j in range(NUM_AREA)}
        self.max_index_used = 0
        self.start_time = time.time()

        # Initialize parameters for policies
        self.prioritization_parameters = {
            i: {j: 0.0 for j in range(len(vehicles))}
            for i in range(NUM_AREA)
        }
        self.mean_response_times = {
            i: {j: 0.0 for j in range(len(vehicles))}
            for i in range(NUM_AREA)
        }

        # Initialize event logs
        self.event_logs = []

        # Set up parameters needed for policies
        self._compute_policy_parameters()

    def log_event(self, event: Event, available_vehicles_for_log: List[int], 
                 chosen_vehicle_id: int, service_time: float, response_time: float) -> None:
        """
        Record an event to the event log.
        
        Args:
            event: The event to log
            available_vehicles_for_log: Vehicles available at event time
            chosen_vehicle_id: Vehicle that was selected
            service_time: Time required to service the call
            response_time: Time required to respond to the call
        """
        self.event_logs.append(EventLog(
            time_of_event=self.current_time,
            time_of_service=service_time,
            response_time=response_time,
            available_engines=available_vehicles_for_log,
            chosen_engine=chosen_vehicle_id,
            mean_response_times=self.mean_response_times[event.area_id],
            prioritization_parameters=self.prioritization_parameters[event.area_id]
        ))

    def save_logs_to_excel(self, filename: str) -> None:
        """
        Save event logs to an Excel file.
        Args:
            filename: Path to save the Excel file
        """
        df = pd.DataFrame([e.__dict__ for e in self.event_logs])
        df.to_excel(filename, index=False)

    def _compute_policy_parameters(self) -> None:
        """
        Compute parameters required for dispatch policies.
        This method calculates mean response times based on the precomputed random times.
        """
        for area_id in range(NUM_AREA):
            for vehicle_id, vehicle in enumerate(self.vehicles):
                # For Response Time Policy: compute 90th percentile response times
                responses = self.precomputed_times.responses[(area_id, vehicle_id)]
                self.mean_response_times[area_id][vehicle_id] = np.mean(responses)
        
        self.dispatch_policy.compute_parameters(self.precomputed_times, self.vehicles, NUM_AREA, self.mean_response_times)

    def _select_vehicle(self, area_id: int) -> Optional[int]:
        """
        Select a vehicle to dispatch based on the current policy.
        Args:
            area_id: ID of the area requiring service
        Returns:
            ID of the selected vehicle, or None if no vehicle is available
        """
        if not self.available_vehicles:
            return None
        return self.dispatch_policy.select_vehicle(
            area_id, self.available_vehicles)

    def _get_next_time(self, area_id: int, vehicle_id: int, time_type: str) -> float:
        # choose the container and key by type
        if time_type == 'arrival':
            container = self.precomputed_times.arrivals
            key = (area_id, 0)  # single per-area arrival stream
        elif time_type == 'service':
            container = self.precomputed_times.services
            key = (area_id, vehicle_id)
        elif time_type == 'response':
            container = self.precomputed_times.responses
            key = (area_id, vehicle_id)
        else:
            raise ValueError(f"Unknown time type: {time_type}")

        idx = self.time_indices[key]
        self.max_index_used = max(self.max_index_used, idx)

        # running-low check against the correct container
        if idx >= len(container[key]) - 100:
            warnings.warn(f"Running low on {time_type} times for area {area_id}, key {key}")

        self.time_indices[key] += 1
        return container[key][idx]

    def _process_queue(self) -> None:
        """
        Process waiting calls from the queue when vehicles become available.
        """
        available_vehicles_log = list(self.available_vehicles)
        while self.waiting_queue and self.available_vehicles:
            arrival_time, area_id = self.waiting_queue.pop(0)
            chosen_vehicle_id = self._select_vehicle(area_id)

            if chosen_vehicle_id is None:
                self.waiting_queue.insert(0, (arrival_time, area_id))
                break

            self.available_vehicles.remove(chosen_vehicle_id)

            service_time = self._get_next_time(area_id, chosen_vehicle_id, 'service')
            response_time = self._get_next_time(area_id, chosen_vehicle_id, 'response')

            # Log the event
            self.log_event(
                Event(self.current_time, EventType.QUEUE_PROCESSING, area_id),
                available_vehicles_log, 
                chosen_vehicle_id, 
                service_time, 
                response_time
            )

            total_response_time = (self.current_time - arrival_time) + response_time
            self.response_times.append(total_response_time)
            self.total_services += 1

            heapq.heappush(
                self.events,
                Event(self.current_time + service_time, EventType.COMPLETION, 
                      area_id, chosen_vehicle_id)
            )

    def _handle_arrival_event(self, event: Event) -> None:
        """
        Process an arrival event.
        
        Args:
            event: The arrival event to process
        """
        # Schedule the next arrival from this source
        next_arrival = self.current_time + self._get_next_time(
            event.area_id, event.vehicle_id, 'arrival')
        heapq.heappush(
            self.events,
            Event(next_arrival, EventType.ARRIVAL, event.area_id,
                  event.vehicle_id, next_arrival)
        )

        # Handle the current arrival
        service_time = 0
        response_time = 0
        chosen_vehicle_id = None
        
        if self.available_vehicles:
            available_vehicles_for_log = list(self.available_vehicles)
            chosen_vehicle_id = self._select_vehicle(event.area_id)
            
            if chosen_vehicle_id is not None:
                self.available_vehicles.remove(chosen_vehicle_id)
                service_time = self._get_next_time(event.area_id, chosen_vehicle_id, 'service')
                response_time = self._get_next_time(event.area_id, chosen_vehicle_id, 'response')
                
                self.total_service_time += service_time
                self.response_times.append(response_time)
                self.total_services += 1
                
                heapq.heappush(
                    self.events,
                    Event(self.current_time + service_time, EventType.COMPLETION, 
                          event.area_id, chosen_vehicle_id)
                )
        else:
            available_vehicles_for_log = []
            self.waiting_queue.append((event.arrival_time, event.area_id))
            self.max_queue_size = max(self.max_queue_size, len(self.waiting_queue))
        
        # Log the event
        self.log_event(
            event, 
            available_vehicles_for_log, 
            chosen_vehicle_id, 
            service_time, 
            response_time
        )

    def _handle_completion_event(self, event: Event) -> None:
        """
        Process a completion event.
        
        Args:
            event: The completion event to process
        """
        self.available_vehicles.add(event.vehicle_id)
        self._process_queue()

    def run(self, end_time: float) -> None:
        """
        Run the simulation until the specified end time.
        Args:
            end_time: Time at which to stop the simulation
        """
        run_start_time = time.time()

        for area_id in range(NUM_AREA):
            arrival_time = self._get_next_time(area_id, 0, 'arrival')
            heapq.heappush(
                self.events,
                Event(arrival_time, EventType.ARRIVAL, area_id, 0, arrival_time)
            )

        # Process events until end time
        last_update_time = 0
        
        while self.events and self.current_time < end_time:
            event = heapq.heappop(self.events)
            
            # Update progress bar based on current time
            if event.time - last_update_time > end_time / 100:
                last_update_time = event.time
            
            self.current_time = event.time

            # time-weighted queue length (average queue = area under curve / total time)
            dt = event.time - self._last_time_for_avg
            if dt > 0:
                self._queue_area += len(self.waiting_queue) * dt
                self._last_time_for_avg = event.time
            # Process event based on type
            if event.event_type == EventType.ARRIVAL:
                self._handle_arrival_event(event)
            elif event.event_type == EventType.COMPLETION:
                self._handle_completion_event(event)
        # close the integral up to end_time and compute average queue size
        if self._last_time_for_avg < end_time:
            self._queue_area += len(self.waiting_queue) * (end_time - self._last_time_for_avg)
        self.avg_queue_size = self._queue_area / max(end_time, 1e-9)
        logger.info(f"Simulation completed in {time.time() - run_start_time:.2f} seconds")
     
        
