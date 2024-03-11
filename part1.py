import pandas as pd
from sklearn.model_selection import KFold
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

from preprocess import preprocess_data

class WaterPumpPredictor:
    def __init__(self, model_type, numerical_preprocessing, categorical_preprocessing, processed_data):
        self.model_type = model_type
        self.numerical_preprocessing = numerical_preprocessing
        self.categorical_preprocessing = categorical_preprocessing
        self.processed_data = processed_data
        self.numerical_features = []
        self.categorical_feature = []
        self.pipeline = self._build_pipeline()
        
    def _select_model(self):
        models = {
            'RandomForestClassifier': RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, n_jobs=-1),
            'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8),
            'HistGradientBoostingClassifier': HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_depth=10, min_samples_leaf=20),
            'LogisticRegression': LogisticRegression(max_iter=500, penalty='l2', C=1.0, solver='lbfgs'),
            'MLPClassifier': MLPClassifier(max_iter=400, hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, learning_rate='constant')
        }
        
        return models.get(self.model_type, models['RandomForestClassifier'])
        
    def _build_pipeline(self):
        #automatically finding the numerical and categorical features
        self.numerical_features = self.processed_data.select_dtypes(exclude=['object', 'bool']).columns.drop(['id']).tolist()
        self.categorical_features = self.processed_data.select_dtypes(include=['object', 'bool']).columns.drop('status_group').tolist()
        
        #preprocessing pipelines
        numerical_transformers = []
        #fill in missing values
        numerical_transformers.append(('imputer', SimpleImputer(strategy='mean')))

        #optional scaling
        if self.numerical_preprocessing == 'StandardScaler':
            numerical_transformers.append(('scaler', StandardScaler()))

        categorical_transformers = []

        def convert_to_string(X):
            return X.astype(str)
        
        convert_to_string_transformer = FunctionTransformer(convert_to_string)

        categorical_transformers.append(('to_string', convert_to_string_transformer))
        categorical_transformers.append(('imputer', SimpleImputer(strategy='constant', fill_value='missing')))

        if self.categorical_preprocessing == 'OneHotEncoder':
            if self.model_type == 'HistGradientBoostingClassifier': #requires dense matrix
                categorical_transformers.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
            else:
                categorical_transformers.append(('encoder', OneHotEncoder(handle_unknown='ignore')))
        elif self.categorical_preprocessing == 'OrdinalEncoder':
            categorical_transformers.append(('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))
        elif self.categorical_preprocessing == 'TargetEncoder':
            categorical_transformers.append(('encoder', TargetEncoder()))


        preprocessor = ColumnTransformer(transformers=[
            ('num', Pipeline(numerical_transformers), self.numerical_features),
            ('cat', Pipeline(categorical_transformers), self.categorical_features)
        ])

        model = self._select_model() #defaults to random forest classifier

        #selectfrommodel selects features based on their importance weights, (detrmined by estimator passed in)
        #log regression estimator, uses l1 to make more sparsity in co-efficients, prune coefficients that are less important (0)
        # estimator = SelectFromModel(LogisticRegression(max_iter=3000, penalty='l1', solver='liblinear'))

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            # ('feature_selection', estimator),                       
            ('model', model)])
        
        return pipeline
          
    def train_and_evaluate(self, X, y):
        #k-fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        start_time = time.time()

        scores = []
        for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            fold_start_time = time.time()
            
            #fit the model 
            self.pipeline.fit(X_train, y_train)
            
            # Predict and evaluate
            predictions = self.pipeline.predict(X_test)
            score = accuracy_score(y_test, predictions)
            scores.append(score)
            
            fold_end_time = time.time()
            
            fold_elapsed_time = fold_end_time - fold_start_time
            
            #printing the progress
            print(f"Fold {fold}: Accuracy = {score:.4f}")
            print(f"Elapsed time for current fold: {fold_elapsed_time:.2f} seconds")

        #printing overall performance
        print(f"Average Cross-Validation Score: {np.mean(scores):.4f}")
        return scores
    
    def predict_and_save(self, X, output_file):
        #predict on the test set
        test_data = preprocess_data(X)
        X_test = test_data.drop('id', axis=1)
        test_predictions = self.pipeline.predict(X_test)
        
        output_df = pd.DataFrame({'id': X['id'],'status_group': test_predictions})
        
        output_df.to_csv(output_file, index=False)      

def main():
    #load and preprocess data
    train_values = pd.read_csv('data/training_set_values.csv')
    train_labels = pd.read_csv('data/training_set_labels.csv')
    test_values = pd.read_csv('data/test_set_values.csv')
    test_prediction_output_file = 'abcd.csv'
    
    #merge datasets on 'id'
    data = train_values.merge(train_labels, on='id')

    processed_data = preprocess_data(data)
    
    #initialise predictor
    predictor = WaterPumpPredictor(
        model_type='GradientBoostingClassifier',
        numerical_preprocessing='StandardScaler',
        categorical_preprocessing='OneHotEncoder',
        processed_data=processed_data,
    )
    
    
    
    X_train = processed_data.drop(['id', 'status_group'], axis=1)
    y_train = processed_data['status_group']

    #training and evaluation
    scores = predictor.train_and_evaluate(X_train, y_train)
    
    #predict and save results
    predictor.predict_and_save(test_values, test_prediction_output_file)
    
    
if __name__ == '__main__':
    main()
