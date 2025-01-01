import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
import numpy as np

# Load the dataset from a CSV file
data = pd.read_csv('heart_failure.csv')

# Display information about the dataset, including column names and data types
print(data.info())

# Show the distribution of the target variable ('death_event')
print('Collections and number of values from the dataset: ', Counter(data['death_event']))

# Target variable (outcome indicating death event)
y = data['death_event']

# Features (predictors)
x = data[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
          'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium',
          'sex', 'smoking', 'time']]

# Convert categorical variables into dummy/indicator variables
x = pd.get_dummies(x)

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Apply standard scaling to numeric columns to normalize them
ct = ColumnTransformer([('numeric', StandardScaler(), 
                         ['age', 'creatinine_phosphokinase', 'ejection_fraction', 
                          'platelets', 'serum_creatinine', 'serum_sodium', 'time'])])

# Fit the scaler on the training data and transform it
X_train = ct.fit_transform(X_train)

# Transform the testing data using the same scaler
X_test = ct.transform(X_test)

# Encode target labels (death_event) as integers
le = LabelEncoder()

Y_train = le.fit_transform(Y_train.astype(str))  # Convert to string before encoding
Y_test = le.transform(Y_test.astype(str))

# Convert the integer-encoded labels to one-hot encoded format for classification
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

# Define the sequential model
model = Sequential()

# Add an input layer with the same shape as the number of features
model.add(InputLayer(input_shape=(X_train.shape[1], )))

# Add a dense (fully connected) layer with 12 neurons and ReLU activation
model.add(Dense(12, activation='relu'))

# Add an output layer with 2 neurons (for binary classification) and softmax activation
model.add(Dense(2, activation='softmax'))

# Compile the model with categorical crossentropy loss, Adam optimizer, and accuracy as a metric
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Train the model for 100 epochs with a batch size of 16
model.fit(X_train, Y_train, epochs=100, batch_size=16)

# Evaluate the model on the test set
loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print(f"Model Loss: {loss}, Model Accuracy: {acc}")

# Predict the class probabilities for the test set
y_estimate = model.predict(X_test)

# Convert probabilities to class predictions
y_estimate = np.argmax(y_estimate, axis=1)

# Get the true labels from the one-hot encoded format
y_true = np.argmax(Y_test, axis=1)

# Print a classification report to evaluate model performance
print(classification_report(y_true, y_estimate))
