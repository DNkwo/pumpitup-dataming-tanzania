import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import time


train_input_file, train_labels_file, test_input_file, numerical_preprocessing, categorical_preprocessing, model_type, test_prediction_output_file = sys.argv[1:]

# numerical_preprocessing = 'StandardScaler'
# categorical_preprocessing = 'OrdinalEncoder'
# model_type = 'MLPClassifier'

# Load the datasets
train_values = pd.read_csv('data/training_set_values.csv')
train_labels = pd.read_csv('data/training_set_labels.csv')
test_values = pd.read_csv('data/test_set_values.csv')


pd.set_option('display.max_columns', None)

# Merge datasets on 'id'
data = train_values.merge(train_labels, on='id')


# # Function to reduce cardinality

def reduce_cardinality(df, column, n=20):

   # Get the top n most frequent categories
    top_n_categories = df[column].value_counts().nlargest(n).index
    
    # Replace categories not in the top n with 'Other'
    df[column] = df[column].where(df[column].isin(top_n_categories), other='Other')
    
    return df

#reducing cardinalities in these
for feature in ['funder', 'installer', 'scheme_name', 'ward', "lga"]:
    data = reduce_cardinality(data, feature)
    
#dropping more categories that do not seem so useful (e.g too many missing features, not relevant)
#wpt_name - doubt arbritary names will affect predictive power
#subvillage - doubt arbritary names will affect predictive power
#recorded_by - mostly identical rows, probably wont affect predictive power
#num_private - all identical rows
#waterpoint_type_group - basically the same as 'waterpoint_type'
#quantity_group - basically the same as 'quantity'
#payment_type - basically the same as 'payment'
#extraction_type_group - basically same as 'extraction_type'
#water_quality - redundant as 'quality_group' is just a generalised form, mostly the same, so can be removed
data.drop(['wpt_name', 'subvillage', "recorded_by",
           "num_private", "waterpoint_type_group", "quantity_group", "payment_type",
           "extraction_type_group", "water_quality"], axis=1, inplace=True)


# Preprocess 'date_recorded' into more usable features
data['date_recorded'] = pd.to_datetime(data['date_recorded'])

#probably sufficient to only include the year recorded, we reduce features this way
data['year_recorded'] = data['date_recorded'].dt.year
# data['month_recorded'] = data['date_recorded'].dt.month
# data['day_recorded'] = data['date_recorded'].dt.day

# Drop original 'date_recorded' column
data.drop('date_recorded', axis=1, inplace=True)



# Automatically identify feature types
categorical_features = data.select_dtypes(include=['object', 'bool']).columns.drop('status_group')
# numerical_features = data.select_dtypes(exclude=['object', 'bool']).columns.drop(['id', 'year_recorded', 'month_recorded', 'day_recorded'])
numerical_features = data.select_dtypes(exclude=['object', 'bool']).columns.drop(['id', 'year_recorded'])

cardinality = data[categorical_features].nunique()
print(cardinality)

#correlation-based feature elimination on numerical features
corr_matrix = data[numerical_features].corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# Drop features 
data = data.drop(to_drop, axis=1)


#preprocessing pipelines
numerical_transformers = []

#fill in missing values
numerical_transformers.append(('imputer', SimpleImputer(strategy='mean')))

#optional scaling
if numerical_preprocessing == 'StandardScaler':
    numerical_transformers.append(('scaler', StandardScaler()))



def convert_to_string(X):
    return X.astype(str)

categorical_transformers = []

convert_to_string_transformer = FunctionTransformer(convert_to_string)

categorical_transformers.append(('to_string', convert_to_string_transformer))
categorical_transformers.append(('imputer', SimpleImputer(strategy='constant', fill_value='missing')))



if categorical_preprocessing == 'OneHotEncoder':
    if model_type == 'HistGradientBoostingClassifier': #requires dense matrix
        categorical_transformers.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
    else:
        categorical_transformers.append(('encoder', OneHotEncoder(handle_unknown='ignore')))
elif categorical_preprocessing == 'OrdinalEncoder':
    categorical_transformers.append(('encoder', OrdinalEncoder()))
elif categorical_preprocessing == 'TargetEncoder':
    categorical_transformers.append(('encoder', TargetEncoder()))


preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline(numerical_transformers), numerical_features),
    ('cat', Pipeline(categorical_transformers), categorical_features)
])


# Split features and labels
X = data.drop(['id', 'status_group'], axis=1)
y = data['status_group']

models = {
    'RandomForestClassifier': RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, n_jobs=-1),
    'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8),
    'HistGradientBoostingClassifier': HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_depth=10, min_samples_leaf=20),
    'LogisticRegression': LogisticRegression(max_iter=1000, penalty='l2', C=1.0, solver='lbfgs'),
    'MLPClassifier': MLPClassifier(max_iter=400, hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, learning_rate='constant')
}


model = models.get(model_type, RandomForestClassifier(n_jobs=-1)) #defaults to random forest classifier

#selectfrommodel selects features based on their importance weights, (detrmined by estimator passed in)
#log regression estimator, uses l1 to make more sparsity in co-efficients, prune coefficients that are less important (0)
# estimator = SelectFromModel(LogisticRegression(max_iter=3000, penalty='l1', solver='liblinear'))

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    # ('feature_selection', estimator),                       
    ('model', model)])

# K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

start_time = time.time()

scores = []
for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    fold_start_time = time.time()
    # Fit the model (ensure your model pipeline is defined as 'model')
    
    
    pipeline.fit(X_train, y_train)
    
    # Predict and evaluate
    predictions = pipeline.predict(X_test)
    score = accuracy_score(y_test, predictions)
    scores.append(score)
    
    fold_end_time = time.time()
    
    fold_elapsed_time = fold_end_time - fold_start_time
    
    # Print the progress
    print(f"Fold {fold}: Accuracy = {score:.4f}")
    print(f"Elapsed time for current fold: {fold_elapsed_time:.2f} seconds")

# Print overall performance
print(f"Average Cross-Validation Score: {np.mean(scores):.4f}")

