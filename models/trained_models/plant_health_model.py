import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import mkl
# Load the dataset
data = pd.DataFrame({
    'Soil_Moisture': [42, 21, 31, 53, 38, 49, 29, 43, 34, 51, 30],
    'Temperature': [26, 36, 31, 27, 23, 29, 24, 38, 33, 37, 25],
    'Nutrient_Levels': [5, 4, 7, 8, 6, 3, 5, 2, 9, 4, 8],
    'Acidity': [6.8, 6.6, 6.3, 6.7, 6.1, 6.5, 6.9, 7.2, 7.0, 7.1, 6.4],
    'Pest_Activity': [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
    'Oxygen_Levels': [24, 21, 19, 22, 26, 18, 23, 20, 25, 17, 24],
    'Manure_Requirements': [4, 3, 5, 6, 4, 5, 4, 6, 3, 5, 4],
    'Weed_Presence': [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
    'Plant_Health': [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0]
})

# Split data into features (X) and target variable (y)
X = data.drop('Plant_Health', axis=1).values
y = data['Plant_Health'].values

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Define neural network model
class PlantHealthModel(nn.Module):
    def __init__(self):
        super(PlantHealthModel, self).__init__()
        self.fc1 = nn.Linear(8, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Instantiate the model
model = PlantHealthModel()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

# Evaluate the model
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test).sum().item() / len(y_test)
    print('Accuracy:', accuracy)
