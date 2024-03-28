import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
import onemkl

class PlantHealthPredictor:
    def __init__(self):
        self.model = None

    def generate_data(self, num_samples=1000):
        # Generate synthetic data for plant health indicators
        soil_moisture = np.random.uniform(0, 100, num_samples)
        temperature = np.random.uniform(10, 40, num_samples)
        nutrient_levels = np.random.uniform(0, 10, num_samples)
        pest_activity = np.random.randint(0, 2, num_samples)  # Binary indicator for pest activity
        oxygen_levels = np.random.uniform(10, 30, num_samples)
        manure_requirements = np.random.uniform(0, 10, num_samples)
        weed_presence = np.random.randint(0, 2, num_samples)  # Binary indicator for weed presence

        # Define plant health labels based on indicators
        plant_health = np.where((soil_moisture > 50) & (temperature < 30) & (nutrient_levels > 5) &
                                (pest_activity == 0) & (oxygen_levels > 20) & (manure_requirements > 3) &
                                (weed_presence == 0), 1, 0)

        return np.column_stack((soil_moisture, temperature, nutrient_levels, pest_activity, 
                                oxygen_levels, manure_requirements, weed_presence)), plant_health

    def train_model(self, X, y):
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define neural network model
        self.model = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Output layer with 2 units for binary classification
        )

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Train the model
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = self.model(torch.tensor(X_train, dtype=torch.float32))
            loss = criterion(outputs, torch.tensor(y_train))
            loss.backward()
            optimizer.step()

    def evaluate_model(self, X_test, y_test):
        # Evaluate model on test data
        outputs = self.model(torch.tensor(X_test, dtype=torch.float32))
        _, predicted = torch.max(outputs, 1)
        y_pred = predicted.numpy()
        report = classification_report(y_test, y_pred)
        print("Classification Report:\n", report)

    def predict(self, sample):
        # Predict plant health for a new sample
        output = self.model(torch.tensor(sample, dtype=torch.float32))
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

        # Soil Moisture
        if sample[0] < 30:
            optimal_actions.append("Increase soil moisture by watering.")

        # Temperature
        if sample[1] > 30:
            optimal_actions.append("Provide shade or reduce exposure to direct sunlight.")

        # Nutrient Levels
        if sample[2] < 3:
            optimal_actions.append("Apply fertilizer or nutrient supplements.")

        # Pest Activity
        if sample[3] == 1:
            optimal_actions.append("Implement pest control measures.")

        # Oxygen Levels
        if sample[4] < 20:
            optimal_actions.append("Improve ventilation or aeration.")

        # Manure Requirements
        if sample[5] < 5:
            optimal_actions.append("Increase organic matter or apply compost.")

        # Weed Presence
        if sample[6] == 1:
            optimal_actions.append("Remove weeds manually or use herbicides.")

        print("Optimal resource allocation:")
        for action in optimal_actions:
            print("-", action)

# Example usage:
if __name__ == "__main__":
    # Create plant health predictor
    predictor = PlantHealthPredictor()

    # Generate synthetic data
    X, y = predictor.generate_data()

    # Train model
    predictor.train_model(X, y)

    # Get user input for plant health indicators
    soil_moisture = float(input("Enter soil moisture value (0-100): "))
    temperature = float(input("Enter temperature value (10-40): "))
    nutrient_levels = float(input("Enter nutrient levels value (0-10): "))
    pest_activity = int(input("Enter pest activity (0: no activity, 1: activity): "))
    oxygen_levels = float(input("Enter oxygen levels value (10-30): "))
    manure_requirements = float(input("Enter manure requirements value (0-10): "))
    weed_presence = int(input("Enter weed presence (0: no weeds, 1: weeds present): "))
    sample = np.array([[soil_moisture, temperature, nutrient_levels, pest_activity, 
                        oxygen_levels, manure_requirements, weed_presence]])

    # Predict plant health
    prediction = predictor.predict(sample)

    # Provide feedback based on plant health prediction
    predictor.feedback(prediction)

    # Optimal resource allocation
    predictor.optimal_resource_allocation(sample[0])

