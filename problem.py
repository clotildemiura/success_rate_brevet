import os
import numpy as np
import pandas as pd
import rampwf as rw
from rampwf.workflows import FeatureExtractorRegressor
from rampwf.score_types.base import BaseScoreType
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error


problem_title = 'Prediction of succes rates at the Brevet exam'
_target_column_name = 'target' 
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression()
# An object implementing the workflow

class BREVET(FeatureExtractorRegressor):
    def __init__(self, workflow_element_names=[
            'feature_extractor', 'regressor', 'cities_data_filtered.csv']):
        super(BREVET, self).__init__(workflow_element_names[:2])
        self.element_names = workflow_element_names

workflow = BREVET()

# define the score 
class NormalizedRMSE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='rmse', precision=5):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return rmse / np.std(y_true) 
    
score_types = [
    NormalizedRMSE(name='normalized rmse', precision=5),
]

def get_cv(X, y):
    cv = ShuffleSplit(n_splits=8, test_size=0.20, random_state=42)
    return cv.split(X, y)

def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name), index_col=0)
    y_array = data[_target_column_name].values
    X_df = data.drop(_target_column_name, axis=1)
    return X_df, y_array

def get_train_data(path='.'):
    f_name = 'data_college_filtered_TRAIN.csv'
    return _read_data(path, f_name)

def get_test_data(path='.'):
    f_name = 'data_college_filtered_TEST.csv'
    return _read_data(path, f_name)