# main.py

from src._plant_health_data import PlantHealthPredictor
import numpy as np

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
