import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch.nn as nn
import mkl

try:
    from torch.utils import mkl as mkl_torch
except ImportError:
    mkl_torch = None

# Set the number of threads to be used by MKL for parallel execution
mkl.set_num_threads(4)  # Adjust the number of threads as needed

class PlantHealthPredictor:
    def __init__(self):
        self.model = None

    def load_data_from_file(self, file_path):
        # Load data from CSV file
        data = pd.read_csv(file_path)
        X = data.iloc[:, :-1].values  # Features (all columns except the last)
        y = data.iloc[:, -1].values    # Labels (last column)
        return X, y

    def train_model(self, X, y):
        # Convert labels to Long data type
        y_train = torch.tensor(y, dtype=torch.long)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_train, test_size=0.2, random_state=42)

        # Define neural network model
        self.model = nn.Sequential(
            nn.Linear(X.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Output layer with 2 units for binary classification
        )

        # Use MKL for PyTorch if available
        if mkl_torch is not None:
            self.model = self.model.to_mkldnn()

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Train the model
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = self.model(torch.tensor(X_train, dtype=torch.float32))
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

    def predict(self, sample):
        # Predict plant health for a new sample
        output = self.model(torch.tensor(sample, dtype=torch.float32).unsqueeze(0))  # Add unsqueeze(0) to add batch dimension
        _, predicted = torch.max(output, 1)
        return predicted.item()

    def feedback(self, prediction):
        # Provide feedback based on plant health prediction
        if prediction == 1:
            print("The plant is healthy. Optimal conditions observed.")
        else:
            print("The plant is unhealthy. Attention needed for optimal growth.")

    def optimal_resource_allocation(self, sample):
        # Optimal resource allocation based on plant health indicators
        optimal_actions = []

        # Sensor values in the CSV
        sensor_names = ["Soil Moisture", "Temperature", "Nutrient Levels", "Acidity (pH)",
                        "Pest Activity", "Oxygen Levels", "Manure Requirements", "Weed Presence"]

        # Print sensor values used for prediction
        print("Sensor values used for prediction:")
        for i, value in enumerate(sample):
            print(f"{sensor_names[i]}: {value}")

        # Soil Moisture
        if sample[0] < 30:
            optimal_actions.append("Increase soil moisture by watering.")

        # Temperature
        if sample[1] > 30:
            optimal_actions.append("Provide shade or reduce exposure to direct sunlight.")

        # Nutrient Levels
        if sample[2] < 3:
            optimal_actions.append("Apply fertilizer or nutrient supplements.")

        # Acidity (pH)
        if sample[3] < 6:
            optimal_actions.append("Adjust pH level by adding lime or other amendments.")

        # Pest Activity
        if sample[4] == 1:
            optimal_actions.append("Implement pest control measures.")

        # Oxygen Levels
        if sample[5] < 20:
            optimal_actions.append("Improve ventilation or aeration.")

        # Manure Requirements
        if sample[6] < 5:
            optimal_actions.append("Increase organic matter or apply compost.")

        # Weed Presence
        if sample[7] == 1:
            optimal_actions.append("Remove weeds manually or use herbicides.")

        print("\nOptimal resource allocation:")
        if len(optimal_actions) == 0:
            print("No specific actions recommended based on the current conditions.")
        else:
            for action in optimal_actions:
                print("-", action)

if __name__ == "__main__":
    # Create plant health predictor
    predictor = PlantHealthPredictor()

    # Load data from file
    X, y = predictor.load_data_from_file("data.csv")

    # Train model
    predictor.train_model(X, y)

    # Sample a random input for prediction
    random_index = np.random.randint(len(X))
    sample = X[random_index]

    # Predict plant health
    prediction = predictor.predict(sample)

    # Provide feedback based on plant health prediction
    predictor.feedback(prediction)

    # Optimal resource allocation
    predictor.optimal_resource_allocation(sample)
