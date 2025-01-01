# Heart Failure Prediction Using Machine Learning

## Project Summary
In this project, a machine learning model is developed to predict the survival of patients with heart failure. The dataset used is sourced from Kaggle and contains 12 features, including:

- **Serum creatinine**
- **Ejection fraction**
- **Age**
- **Anemia**
- **Diabetes**
- **High blood pressure**
- Other relevant health indicators

### Steps Involved:
1. **Dataset Loading**:
   - The dataset is loaded using Pandas, and an overview of its structure is obtained using `data.info()`.

2. **Exploratory Analysis**:
   - The distribution of the target variable (`death_event`) is examined using the `Counter` class.

3. **Feature Selection**:
   - Relevant features for the prediction are selected, and categorical variables are converted to dummy/indicator variables using `pd.get_dummies`.

4. **Data Splitting**:
   - The dataset is split into training (70%) and testing (30%) sets using `train_test_split`.

5. **Data Preprocessing**:
   - Numerical features are scaled using `StandardScaler` to normalize the data.
   - The target variable is label-encoded and converted to a one-hot encoded format for the classification task.

6. **Model Definition**:
   - A sequential neural network model is built using TensorFlow/Keras with the following layers:
     - Input layer matching the number of features.
     - Dense hidden layer with 12 neurons and ReLU activation.
     - Output layer with 2 neurons (for binary classification) and softmax activation.

7. **Model Training**:
   - The model is compiled with categorical crossentropy loss, Adam optimizer, and accuracy as a metric.
   - It is trained over 100 epochs with a batch size of 16.

8. **Evaluation**:
   - The model is evaluated on the test set, and metrics such as loss and accuracy are computed.
   - Predictions are made, and a classification report is generated to assess performance.

---

## Theory
Cardiovascular diseases (CVDs) are the leading cause of death globally, accounting for approximately 17.9 million deaths annually, which represents 31% of all global deaths. Heart failure, often a result of CVDs, is a major health concern. This dataset contains features that can help predict mortality due to heart failure.

### Key Points:
- **Preventability**:
  - Many CVDs can be prevented by addressing behavioral risk factors such as:
    - Tobacco use
    - Unhealthy diet and obesity
    - Physical inactivity
    - Harmful alcohol use

- **Need for Early Detection**:
  - Early detection and management of high-risk individuals with factors like hypertension, diabetes, hyperlipidemia, or existing disease are crucial.

### Role of Machine Learning:
Machine learning models, like the one implemented in this project, provide significant support in identifying high-risk patients. By leveraging predictive analytics, healthcare professionals can make informed decisions and implement timely interventions to reduce mortality rates.

