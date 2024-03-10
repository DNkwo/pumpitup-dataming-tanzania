import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score
import numpy as np

# Load the datasets
values_df = pd.read_csv('data/training_set_values.csv')
labels_df = pd.read_csv('data/training_set_labels.csv')

pd.set_option('display.max_columns', None)

# Merge datasets on 'id'
data = values_df.merge(labels_df, on='id')

# Preprocess 'date_recorded' into more usable features
data['date_recorded'] = pd.to_datetime(data['date_recorded'])
data['year_recorded'] = data['date_recorded'].dt.year
data['month_recorded'] = data['date_recorded'].dt.month
data['day_recorded'] = data['date_recorded'].dt.day

# Drop original 'date_recorded' column
data.drop('date_recorded', axis=1, inplace=True)

# Automatically identify feature types
categorical_features = data.select_dtypes(include=['object', 'bool']).columns.drop('status_group')
numerical_features = data.select_dtypes(exclude=['object', 'bool']).columns.drop(['id', 'year_recorded', 'month_recorded', 'day_recorded'])


# Create preprocessing pipelines
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

def convert_to_string(X):
    return X.astype(str)

convert_to_string_transformer = FunctionTransformer(convert_to_string)

categorical_pipeline = Pipeline([
    ('to_string', convert_to_string_transformer),
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Define the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier())
])

# Split features and labels
X = data.drop(['id', 'status_group'], axis=1)
y = data['status_group']

print(X.columns.to_list())

# K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

# print(f'Cross-validation scores: {cv_scores}')
# print(f'Average accuracy: {cv_scores.mean()}')

scores = []
for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Fit the model (ensure your model pipeline is defined as 'model')
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    predictions = model.predict(X_test)
    score = accuracy_score(y_test, predictions)
    scores.append(score)
    
    # Print the progress
    print(f"Fold {fold}: Accuracy = {score:.4f}")

# Print overall performance
print(f"Average Cross-Validation Score: {np.mean(scores):.4f}")