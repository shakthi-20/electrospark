##This python code can be used to generate data

import pandas as pd
import random

# Generate random data
data = {
    "Soil Moisture": [random.randint(0, 100) for _ in range(100)],
    "Temperature": [random.randint(10, 40) for _ in range(100)],
    "Nutrient Levels": [random.randint(0, 5) for _ in range(100)],
    "Acidity (pH)": [random.uniform(4, 8) for _ in range(100)],
    "Pest Activity": [random.randint(0, 10) for _ in range(100)],
    "Oxygen Levels": [random.randint(10, 25) for _ in range(100)],
    "Manure Requirements": [random.randint(0, 5) for _ in range(100)],
    "Weed Presence": [random.randint(0, 1) for _ in range(100)]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("data.csv", index=False)
