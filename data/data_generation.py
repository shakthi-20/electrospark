##This python code can be used to generate data

import pandas as pd
import random

# Generate random data
data = {
    "Soil Moisture": [random.randint(0, 100) for _ in range(50)],
    "Temperature": [random.randint(10, 40) for _ in range(50)],
    "Nutrient Levels": [random.randint(0, 5) for _ in range(50)],
    "Acidity (pH)": [random.uniform(4, 8) for _ in range(50)],
    "Pest Activity": [random.randint(0, 10) for _ in range(50)],
    "Oxygen Levels": [random.randint(10, 25) for _ in range(50)],
    "Manure Requirements": [random.randint(0, 5) for _ in range(50)],
    "Weed Presence": [random.randint(0, 1) for _ in range(50)]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("data.csv", index=False)
