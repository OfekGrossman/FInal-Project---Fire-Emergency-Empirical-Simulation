# Constants for simulation
NUM_PARAMETER_SETS = 1
NUM_REPLICATIONS = 30
NUM_AREA = 5

SIMULATION_TIME = 1000000
NUM_SAMPLES = 50000
RANDOM_SEED = 42

#score
EPSILON = 0.001

# Parameter ranges
MAX_INTERARRIVAL_RANGE = (500,2000)#(500,10000) full
MAX_TOTAL_SERVICE_RANGE = (100,200)#(10,500) full
STEP_INNTERVAL = 200
STEP_TOTAL_SERVICE = 10
SERVICE_RT_RATIO = 0.75

#RESPONSE_RANGE = (4,6) #ofek para (4,6) || yossi param (1, 6)
UTILIZATION_THRESHOLD = 0.7
CV_SERVICE_RANGE = (0.1, 0.5)
CV_RESPONSE_RANGE = (0.2, 0.6)

EMPERICAL_FILE = 'fire_emergency_OG/empirical_pmf.csv'


# Functions to calculate derived parameters
def calculate_required_samples():
    """Calculate the minimum number of samples needed based on parameters."""
    avg_interarrival = sum(MAX_INTERARRIVAL_RANGE) / 2
    expected_arrivals = SIMULATION_TIME / avg_interarrival
    total_events_per_pair = expected_arrivals * 3  # arrivals, service, response
    total_events = total_events_per_pair * NUM_AREA * NUM_AREA
    return int(total_events * 1.2)  # Add 20% safety margin