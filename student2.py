import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset (replace 'your_file.csv' with the actual file path)
data = pd.read_csv('student/stu dent-mat.csv', sep=';')

# Select features and target
categorical_features = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian',
                        'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
numeric_features = ['age', 'Medu', 'Fedu', 'studytime', 'failures', 'famrel']  # Add the numeric attributes you want
target = 'G3'

# Convert categorical features using one-hot encoding
data_encoded = pd.get_dummies(data, columns=categorical_features, drop_first=True)

# Split data into features (X) and target (y)
X = data_encoded[numeric_features + data_encoded.columns.tolist()]
y = data_encoded[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Predict student performance on the testing set
y_pred = model.predict(X_test)

accuracy = model.score(X_test, y_test)

# Evaluate the model using Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')

# Visualize the predicted vs. actual performance
# plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual G3')
plt.ylabel('Predicted G3')
plt.title('Actual vs. Predicted Student Performance (G3)')
plt.show()


print(f'Model Accuracy: {accuracy * 100:.2f}%')