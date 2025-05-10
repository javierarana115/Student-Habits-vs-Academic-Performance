'''

import kagglehub

# Download latest version
path = kagglehub.dataset_download("jayaantanaath/student-habits-vs-academic-performance")

print("Path to dataset files:", path)

'''
# All Imports
import os
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Loading Dataframe
load_dotenv()
csv_url = os.getenv("CSV_URL")
full_csv_path = os.path.join(csv_url, "student_habits_performance.csv")
df = pd.read_csv(full_csv_path)

# Dropping NaN values because they are irrelevant
df = df.drop(columns=['student_id'])
cleaned_df = df.dropna()
print(cleaned_df.info())
categorical_cols = [
    'parental_education_level',
    'gender',
    'part_time_job',
    'extracurricular_participation',
    'diet_quality',
    'internet_quality'
]

encoded_df = pd.get_dummies(cleaned_df, columns=categorical_cols, dtype=int, drop_first=True)





print(encoded_df.info())

# Ensuring dataframe with only numerical values for correlation heatmap
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_df = encoded_df.select_dtypes(include=numerics)

correlation = numerical_df.corr()

# Displaying correlation heatmap to see what variables are related
plt.figure(figsize=(12, 8))
sns.heatmap(correlation, cmap='coolwarm', annot=True)
plt.title('Correlation Heatmap for different factors')
plt.show()


# Splitting test and train data
X = encoded_df.drop(columns=['exam_score'])
y = encoded_df['exam_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print(f"Training features: {X_train.shape}")
print(f"Test features: {X_test.shape}")
print(f"Training target: {y_train.shape}")
print(f"Test target: {y_test.shape}")

# Using LineearRegression sklearn model to predict test scores
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

importance = pd.DataFrame({
    'Feature' : X_train.columns,
    'Importance' : model.coef_
}).sort_values('Importance', ascending=False)

print(importance.head(3))

# Display predictions from LinearRegression model
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3, label='Linear Regression Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Model Performance')
plt.legend()
plt.grid(True)
plt.show()