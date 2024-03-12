import pandas as pd
from sklearn.model_selection import KFold, train_test_split
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

#HPO
import optuna

from preprocess import preprocess_data

class WaterPumpPredictor:
    def __init__(self, model_type, numerical_preprocessing, categorical_preprocessing, processed_data=None):
        self.model_type = model_type
        self.numerical_preprocessing = numerical_preprocessing
        self.categorical_preprocessing = categorical_preprocessing
        self.processed_data = processed_data
        self.numerical_features = []
        self.categorical_features = []

    def _select_model(self, trial):
        
        model = None
        if trial is not None:
            if self.model_type == 'RandomForestClassifier':
                model = RandomForestClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 100, 500),
                    max_depth=trial.suggest_int('max_depth', 6, 30),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 15),
                    n_jobs=-1
                )
            elif self.model_type == 'GradientBoostingClassifier':
                model = GradientBoostingClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 100, 500),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                    max_depth=trial.suggest_int('max_depth', 3, 10),
                    subsample=trial.suggest_float('subsample', 0.5, 1.0)
                )
            elif self.model_type == 'LogisticRegression':
                model = LogisticRegression(
                    max_iter=trial.suggest_int('max_iter', 100, 1000),
                    penalty=trial.suggest_categorical('penalty', ['l1', 'l2']),
                    C=trial.suggest_float('C', 0.01, 10.0),
                    solver='saga'
                )
            elif self.model_type == 'HistGradientBoostingClassifier':
                model = HistGradientBoostingClassifier(
                    max_iter=trial.suggest_int('max_iter', 200, 1000),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                    max_depth=trial.suggest_int('max_depth', 10, 30),
                    min_samples_leaf=trial.suggest_int('min_samples_split', 20, 30),
                )
            elif self.model_type == 'MLPClassifier':
                model = MLPClassifier(
                    max_iter=trial.suggest_int('max_iter', 200, 1000),
                    hidden_layer_sizes=(trial.suggest_int('hidden_layer', 50, 200),),
                    activation=trial.suggest_categorical('activation', ['tanh', 'relu']),
                    solver=trial.suggest_categorical('solver', ['sgd', 'adam']),
                    alpha=trial.suggest_float('alpha', 0.0001, 0.01),
                    learning_rate=trial.suggest_categorical('learning_rate', ['constant', 'adaptive']),
                )
            else: #just use lgositic regression as an alternative
                model = LogisticRegression(
                    max_iter=trial.suggest_int('max_iter', 100, 1000),
                    penalty=trial.suggest_categorical('penalty', ['l1', 'l2']),
                    C=trial.suggest_float('C', 0.01, 10.0),
                    solver='saga'
                )   
        
        return model
        
    def _build_pipeline(self, trial):
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

        model = self._select_model(trial)

        #selectfrommodel selects features based on their importance weights, (detrmined by estimator passed in)
        #log regression estimator, uses l1 to make more sparsity in co-efficients, prune coefficients that are less important (0)
        # estimator = SelectFromModel(LogisticRegression(max_iter=3000, penalty='l1', solver='liblinear'))

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            # ('feature_selection', estimator),                       
            ('model', model)])
        
        return pipeline
          
    def objective(self, trial):
        #current pipeline of this trial
        pipeline = self._build_pipeline(trial)
        if self.processed_data is None:
            raise ValueError("No preoprocessed data has been provided.")
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.processed_data.drop(['id', 'status_group'], axis=1),
            self.processed_data['status_group'],
            test_size=0.2,
            random_state=42)
        
        
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy
        


def main():
    #load and preprocess data
    train_values = pd.read_csv('data/training_set_values.csv')
    train_labels = pd.read_csv('data/training_set_labels.csv')

    #merge datasets on 'id'
    data = train_values.merge(train_labels, on='id')
    
    processed_data = preprocess_data(data)
    
    #initialise predictor
    predictor = WaterPumpPredictor(
        model_type='LogisticRegression',
        numerical_preprocessing='StandardScaler',
        categorical_preprocessing='OrdinalEncoder',
        processed_data=processed_data,
    )
    
    study = optuna.create_study(direction='maximize')
    study.optimize(predictor.objective, n_trials=50)
    
    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)
    
    
    
if __name__ == '__main__':
    main()
