import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearnex import patch_sklearn
import mkl

# Apply Intel's scikit-learn optimizations
patch_sklearn()

# Set the number of threads to be used by MKL for parallel execution
mkl.set_num_threads(4)  # Adjust the number of threads as needed

class PlantHealthPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None

    def load_data_from_file(self, file_path):
        # Load data from CSV file
        data = pd.read_csv(file_path)
        X = data.iloc[:, :-1].values  # Features (all columns except the last)
        y = data.iloc[:, -1].values    # Labels (last column)
        
        # Normalize features using Min-Max scaling
        self.scaler = MinMaxScaler()
        X_normalized = self.scaler.fit_transform(X)

        return X_normalized, y

    def train_model(self, X, y):
        # Convert labels to Long data type
        y_train = torch.tensor(y, dtype=torch.long)

        # Split data into training and testing sets
        X_train, _, y_train, _ = train_test_split(X, y_train, test_size=0.2, random_state=42)

        # Define neural network model
        self.model = nn.Sequential(
            nn.Linear(X.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Output layer with 2 units for binary classification
        )

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Train the model
        for _ in range(100):
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
                        "Pest Activity", "Oxygen Levels", "Manure Requirements"]

        # Print sensor values used for prediction
        print("Sensor values used for prediction:")
        for i, value in enumerate(sample):
            print(f"{sensor_names[i]}: {value}")

        # Soil Moisture
        if sample[0] < 0.3:
            optimal_actions.append("Increase soil moisture by watering.")

        # Temperature
        if sample[1] > 30:
            optimal_actions.append("Provide shade or reduce exposure to direct sunlight.")

        # Nutrient Levels
        if sample[2] < 0.3:
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
        if sample[6] < 0.5:
            optimal_actions.append("Increase organic matter or apply compost.")

        print("\nOptimal resource allocation:")
        if len(optimal_actions) == 0:
            print("No specific actions recommended based on the current conditions.")
        else:
            for action in optimal_actions:
                print("-", action)
