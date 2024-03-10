import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer #deals with missing values
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


#load from dataset

dataset_values_path = "data/training_set_values.csv"
dataset_labels_path = "data/training_set_labels.csv"
X_dataset = pd.read_csv(dataset_values_path)
y = pd.read_csv(dataset_labels_path)

#merge incase not aligned by index
data = X_dataset.merge(y, on='id')

#split data into features and target variable
X = data.drop(['id'], axis=1)
y = data['status_group']

#train/validation split (80/20)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)


#preprocessing step-----------------------------------------------

#deal with missing values
numerical_imp = SimpleImputer(strategy='mean')
categorical_imp = SimpleImputer(strategy='constant', fill_value='missing')

#identify numerical and categorical columns
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
if 'date_recorded' in categorical_cols:
    categorical_cols = categorical_cols.drop('date_recorded')

print(categorical_cols)

#execute imputation
X_train[numerical_cols] = numerical_imp.fit_transform(X_train[numerical_cols])
X_val[numerical_cols] = numerical_imp.fit_transform(X_val[numerical_cols])

X_train[categorical_cols] = categorical_imp.fit_transform(X_train[categorical_cols]).astype(str)
X_val[categorical_cols] = categorical_imp.fit_transform(X_val[categorical_cols]).astype(str)


#deal with categorical features (OneHotEncoder, Ordinal Encoder and TargetEncoder)

#OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore')
X_train_encoded = pd.DataFrame(encoder.fit_transform(X_train[categorical_cols]))
X_val_encoded = pd.DataFrame(encoder.transform(X_val[categorical_cols]))

print(X_train_encoded.isnull().sum(), X_train_encoded.isnull().sum())


X_train_encoded.index = X_train.index
X_val_encoded.index = X_val.index

# Add one-hot encoded categorical columns to numerical features to reform X_train
num_X_train = X_train.drop(categorical_cols, axis=1)
num_X_val = X_val.drop(categorical_cols, axis=1)
X_train_preprocessed = pd.concat([num_X_train, X_train_encoded], axis=1)
X_val_preprocessed = pd.concat([num_X_val, X_val_encoded], axis=1)

print(X_train_preprocessed.isnull().sum(), X_val_preprocessed.isnull().sum())

#scale numerical values
scaler = StandardScaler()
X_train_preprocessed[numerical_cols] = scaler.fit_transform(X_train_preprocessed[numerical_cols])
X_val_preprocessed[numerical_cols] = scaler.transform(X_val_preprocessed[numerical_cols])


#dealing with datetime features
datetime_col = 'date_recorded'
X_train_preprocessed[datetime_col] = pd.to_datetime(X_train_preprocessed[datetime_col])
X_val_preprocessed[datetime_col] = pd.to_datetime(X_val_preprocessed[datetime_col])

#extract datetime componenents and convert to cyclical features (more useful)

X_train_preprocessed['month'] = X_train_preprocessed[datetime_col].dt.month #convert month as a numerical value
X_val_preprocessed['month'] = X_val_preprocessed[datetime_col].dt.month #convert month as a numerical value


X_train_preprocessed['month_sin'] = np.sin(2 * np.pi * X_train_preprocessed['month'])
X_val_preprocessed['month_sin'] = np.sin(2 * np.pi * X_val_preprocessed['month'])

#then drop original datetime as no longer required
X_train_preprocessed.drop(columns=[datetime_col], inplace=True)
X_val_preprocessed.drop(columns=[datetime_col], inplace=True)


#training step------------------------- (choose between LogisticRegression, RandomForestClassifier, GradientBoostingClaissifer, HistGradientBoostingClassifier and MLPClassifier)
# Initialize classifiers
logistic_regression = LogisticRegression(max_iter=1000)  # Increase max_iter if convergence warnings appear
random_forest = RandomForestClassifier(n_estimators=100)
gradient_boosting = GradientBoostingClassifier(n_estimators=100)
hist_gradient_boosting = HistGradientBoostingClassifier(max_iter=100)
mlp_classifier = MLPClassifier(max_iter=1000)  # Increase max_iter for convergence, especially for large datasets

classifiers = [
    ('Logistic Regression', logistic_regression),
    ('Random Forest', random_forest),
    ('Gradient Boosting', gradient_boosting),
    ('Histogram-based Gradient Boosting', hist_gradient_boosting),
    ('MLP Classifier', mlp_classifier)
]

print(X_train_preprocessed.isnull().sum(), X_val_preprocessed.isnull().sum())
print(np.isinf(X_train_preprocessed).any(), np.isinf(X_val_preprocessed).any())

#evaluation step (do 5-fold cross validation for each combination of dataprocessing and machien)

for name, classifier in classifiers:
    #compute cross-validation scores
    cv_scores = cross_val_score(classifier, X_train_preprocessed, y_train, cv=5, scoring='accuracy')
    
    # Compute the average of the cross-valdiation scores
    cv_accuracy = cv_scores.mean()
    
    print(f'{name} 5-Fold CV Accuracy: {cv_accuracy:.4f}')

#output step