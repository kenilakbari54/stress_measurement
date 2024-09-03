import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv('data/training_data.csv')

# Features and target variable
X = data.drop('stress_level', axis=1)
y = data['stress_level']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
with open('model/stress_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model training complete and saved to 'model/stress_model.pkl'")
