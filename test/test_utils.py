# test_utils.py

import numpy as np

def generate_test_data(num_samples=100):
    """Generate synthetic test data for plant health indicators."""
    soil_moisture = np.random.uniform(0, 100, num_samples)
    temperature = np.random.uniform(10, 40, num_samples)
    nutrient_levels = np.random.uniform(0, 10, num_samples)
    pest_activity = np.random.randint(0, 2, num_samples)
    oxygen_levels = np.random.uniform(10, 30, num_samples)
    manure_requirements = np.random.uniform(0, 10, num_samples)
    weed_presence = np.random.randint(0, 2, num_samples)
    return soil_moisture, temperature, nutrient_levels, pest_activity, oxygen_levels, manure_requirements, weed_presence

def generate_test_sample():
    """Generate a synthetic test sample."""
    sample = np.array([np.random.uniform(0, 100), np.random.uniform(10, 40), np.random.uniform(0, 10),
                       np.random.randint(0, 2), np.random.uniform(10, 30), np.random.uniform(0, 10),
                       np.random.randint(0, 2)])
    return sample

