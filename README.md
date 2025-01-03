# Customer Churn Prediction

This project leverages machine learning to predict customer churn by analyzing customer data. The goal is to identify customers likely to discontinue services, enabling businesses to take proactive measures to improve customer retention.

## Data Preparation

1. **Loaded Data:**
   - Imported the dataset `customer_churn_data.csv` using Pandas:
     ```python
     import pandas as pd
     df = pd.read_csv("customer_churn_data.csv")
     ```

2. **Inspected Dataset:**
   - Used `df.info()`, `df.shape`, and `df.head()` to examine data structure, dimensions, and the first few records.

3. **Handled Missing Values:**
   - Replaced missing values in the `InternetService` column with empty spaces.

4. **Checked for Duplicates:**
   - Verified and removed duplicates using:
     ```python
     df.duplicated().sum()
     ```

---

## Exploratory Data Analysis (EDA)

### Descriptive Statistics
- Calculated descriptive statistics using `df.describe()` to summarize the central tendency, dispersion, and shape of the dataset's distribution.

### Churn Distribution
- Visualized churn distribution using pie charts to identify the proportion of churned and non-churned customers.
  ```python
  df['Churn'].value_counts().plot(kind='pie')
  ```

### Feature Trends
- Analyzed `MonthlyCharges` and `Tenure` using histograms to understand their distributions and detect any skewed patterns.
  ```python
  plt.hist(df['MonthlyCharges'])
  ```
- Explored relationships between `ContractType` and `MonthlyCharges` using bar plots, showing how charges vary by contract type.
  ```python
  df.groupby('ContractType')['MonthlyCharges'].mean().plot(kind="bar")
  ```

### Correlations
- Calculated correlations between numerical columns to identify relationships between variables.
  ```python
  numeric_col_data.corr()
  ```

### Visualizations
-Churn distribution: Pie chart shows the proportion of churned vs. retained customers.

![image](https://github.com/user-attachments/assets/4c6248f9-c177-4853-866a-b338ee6e0871)



-Monthly charges/tenure distributions: Histograms reveal the distribution of these numerical features.

![image](https://github.com/user-attachments/assets/a0ea5ffb-6ee5-4421-a617-3a9199c31eeb)


-Contract type vs. charges: Bar plot illustrates the relationship between contract type and average monthly charges.

![image](https://github.com/user-attachments/assets/c0a246ab-9163-403a-8411-855dbf3fd55c)


## Feature Engineering

1. **Converted Categorical Variables:**
   - Transformed `Gender` and `Churn` columns into numerical values:
     ```python
     df['Gender'] = df['Gender'].apply(lambda x: 1 if x == "Female" else 0)
     df['Churn'] = df['Churn'].apply(lambda x: 1 if x == "Yes" else 0)
     ```

2. **Feature Selection:**
   - Selected relevant features: `Age`, `Gender`, `Tenure`, `MonthlyCharges`.





## Model Training & Evaluation

1. **Data Splitting:**
   - Split dataset into training and testing sets:
     ```python
     from sklearn.model_selection import train_test_split
     X = df[['Age', 'Gender', 'Tenure', 'MonthlyCharges']]
     y = df['Churn']
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     ```

2. **Feature Scaling:**
   - Standardized numerical features using `StandardScaler`:
     ```python
     from sklearn.preprocessing import StandardScaler
     scaler = StandardScaler()
     X_train = scaler.fit_transform(X_train)
     X_test = scaler.transform(X_test)
     ```

======

3. **Model Training**

- **Logistic Regression:**
  ```python
  from sklearn.linear_model import LogisticRegression
  log_model = LogisticRegression()
  log_model.fit(X_train, y_train)
  ```

- **K-Nearest Neighbors:**
  ```python
  from sklearn.neighbors import KNeighborsClassifier
  knn_model = KNeighborsClassifier() # using best parameters from GridSearchCV
  knn_model.fit(X_train, y_train)
  ```

- **Support Vector Machine:**
  ```python
  from sklearn.svm import SVC
  svm_model = SVC() # using best parameters from GridSearchCV
  svm_model.fit(X_train, y_train)
  ```

- **Decision Tree:**
  ```python
  from sklearn.tree import DecisionTreeClassifier
  dt_model = DecisionTreeClassifier() # using best parameters from GridSearchCV
  dt_model.fit(X_train, y_train)
  ```

- **Random Forest:**
  ```python
  from sklearn.ensemble import RandomForestClassifier
  rf_model = RandomForestClassifier() # using best parameters from GridSearchCV
  rf_model.fit(X_train, y_train)
  ```

### Hyperparameter Tuning and Evaluation
- Performed hyperparameter tuning for KNN, SVM, Decision Tree, and Random Forest using `GridSearchCV` to optimize model performance.
- Evaluated model performance using accuracy score:
  ```python
  from sklearn.metrics import accuracy_score
  accuracy = accuracy_score(y_test, model.predict(X_test))
  ```

---

## Model Selection & Persistence

1. **Best Model:**
   - Selected Support Vector Machine (SVC) as it achieved the highest accuracy.

2. **Save Model:**
   - Stored the trained model for future predictions:
     ```python
     import joblib
     joblib.dump(svc_model, "model.pkl")
     ```



## Streamlit Application

A Streamlit app was developed to allow users to input customer details and predict churn likelihood:

```python
import streamlit as st

# Input fields
age = st.number_input('Age', min_value=18, max_value=90, value=40, step=1)
tenure = st.number_input('Tenure', min_value=0, max_value=130, value=10)
monthlycharge = st.number_input('Monthly Charges', min_value=30, max_value=150, value=50)
gender = st.selectbox('Gender', ['Male', 'Female'])

# Button to calculate prediction
calculate = st.button('Calculate')
st.divider()

# Prepare input for model
gender_num = 1 if gender == 'Female' else 0
X = [[age, gender_num, tenure, monthlycharge]]

if calculate:
    st.balloons()
    # Scale input and make prediction
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    
    # Display results
    result = "Yes" if prediction == 1 else "No"
    st.write(f"Prediction: {result}")
else:
    st.write("Please fill in the values and click 'Calculate'")
```

---

##Result

-If the prediction output is 1, the app displays "Prediction: YES," indicating that the customer is likely to churn.
-If the prediction output is 0, the app displays "Prediction: NO," indicating that the customer is not likely to churn.


<img width="797" alt="image" src="https://github.com/user-attachments/assets/d8edf657-2432-4696-93ea-83631984e9c6" />


## Key Findings

- **Influential Factors:** Monthly charges, contract type, and tenure significantly affect customer churn.
- **Best Model:** Support Vector Machine (SVC) outperformed other models in accuracy.
- **Established Workflow:** A complete pipeline was developed for preprocessing, training, evaluation, and deployment of the churn prediction model.

---



