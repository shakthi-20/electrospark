import unittest
import numpy as np
from your_project.src.plant_health_predictor import PlantHealthPredictor

class TestPlantHealthPredictor(unittest.TestCase):
    def setUp(self):
        # Initialize the PlantHealthPredictor object
        self.predictor = PlantHealthPredictor()

    def test_generate_data(self):
        # Test the generate_data method to ensure it returns the correct data shape
        X, y = self.predictor.generate_data(num_samples=100)
        self.assertEqual(X.shape[0], 100)
        self.assertEqual(X.shape[1], 7)  # Ensure correct number of features
        self.assertEqual(y.shape[0], 100)

    def test_train_model(self):
        # Test the train_model method to ensure it trains the model without errors
        X_train = np.random.rand(100, 7)
        y_train = np.random.randint(0, 2, 100)
        self.predictor.train_model(X_train, y_train)
        self.assertIsNotNone(self.predictor.model)

    def test_predict(self):
        # Test the predict method to ensure it returns valid predictions
        sample = np.random.rand(1, 7)
        prediction = self.predictor.predict(sample)
        self.assertIn(prediction, [0, 1])  # Ensure prediction is either 0 or 1

    def test_feedback(self):
        # Test the feedback method to ensure it prints the correct feedback message
        self.assertEqual(self.predictor.feedback(0), "The plant is unhealthy. Attention needed for optimal growth.")
        self.assertEqual(self.predictor.feedback(1), "The plant is healthy. Optimal conditions observed.")

    def test_optimal_resource_allocation(self):
        # Test the optimal_resource_allocation method to ensure it returns valid recommendations
        sample = np.array([[30, 25, 4, 0, 25, 6, 0]])  # Sample input representing unhealthy conditions
        actions = self.predictor.optimal_resource_allocation(sample)
        self.assertIsInstance(actions, list)
        self.assertTrue(all(isinstance(action, str) for action in actions))

if __name__ == '__main__':
    unittest.main()

