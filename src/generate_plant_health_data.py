import csv
import numpy as np

def generate_plant_health_data(file_path, num_samples=1000):
    # Generate synthetic data for plant health indicators
    soil_moisture = np.random.uniform(0, 100, num_samples)
    temperature = np.random.uniform(10, 40, num_samples)
    nutrient_levels = np.random.uniform(0, 10, num_samples)
    pest_activity = np.random.randint(0, 2, num_samples)
    oxygen_levels = np.random.uniform(10, 30, num_samples)
    manure_requirements = np.random.uniform(0, 10, num_samples)
    weed_presence = np.random.randint(0, 2, num_samples)

    # Define plant health labels based on indicators
    plant_health = np.where((soil_moisture > 50) & (temperature < 30) & (nutrient_levels > 5) &
                            (pest_activity == 0) & (oxygen_levels > 20) & (manure_requirements > 3) &
                            (weed_presence == 0), 1, 0)

    # Write data to CSV file
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['soil_moisture', 'temperature', 'nutrient_levels', 'pest_activity',
                         'oxygen_levels', 'manure_requirements', 'weed_presence', 'plant_health'])
        for i in range(num_samples):
            writer.writerow([soil_moisture[i], temperature[i], nutrient_levels[i], pest_activity[i],
                             oxygen_levels[i], manure_requirements[i], weed_presence[i], plant_health[i]])

# Specify the file path where the CSV file will be saved
file_path = 'data/plant_health_data.csv'

# Generate and save the synthetic data
generate_plant_health_data(file_path)

