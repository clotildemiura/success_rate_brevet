import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline 

class CountOrdinalEncoder(OrdinalEncoder):
    """Encode categorical features as an integer array
    usint count information.
    """
    def __init__(self, categories='auto', dtype=np.float64):
        self.categories = categories
        self.dtype = dtype

    def fit(self, X, y=None):
        """Fit the OrdinalEncoder to X.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to determine the categories of each feature.

        Returns
        -------
        self
        """
        super().fit(X)
        X_list, _, _ = self._check_X(X)
        # now we'll reorder by counts
        for k, cat in enumerate(self.categories_):
            counts = []
            for c in cat:
                counts.append(np.sum(X_list[k] == c))
            order = np.argsort(counts)
            self.categories_[k] = cat[order]
        return self

class FeatureExtractor(object):
    def __init__(self):
        
        self.path = os.path.dirname(__file__)
        
        # read the database with the city informations
        self.cities_data = pd.read_csv(os.path.join(self.path, 'cities_data_filtered.csv'), index_col=0)
        self.keep_col_cities = ['population', 'SUPERF', 'med_std_living', 'poverty_rate', 'unemployment_rate']
        
        # Transformers
        
        self.students_col = ['Nb élèves', 'Nb divisions', 'Nb 6èmes 5èmes 4èmes et 3èmes générales',
                             'Nb 6èmes 5èmes 4èmes et 3èmes générales sections européennes et internationales',
                             'Nb 5èmes', 'Nb 4èmes générales', 'Nb 3èmes générales',
                             'Nb 5èmes 4èmes et 3èmes générales Latin ou Grec', 'Nb SEGPA']
        self.students_transformer = FunctionTransformer(self.process_students, validate=False)
        
        self.num_cols = ['Nb élèves', 'Nb 3èmes générales', 'Nb 3èmes générales retardataires',
                         "Nb 6èmes provenant d'une école EP"]
        self.numeric_transformer = Pipeline(steps=[('scale', StandardScaler())])
        
        self.cat_cols = ['Appartenance EP', 'Etablissement sensible', 'CATAEU2010',
                         'Situation relative à une zone rurale ou autre']
        self.categorical_transformer = Pipeline(steps=[('encode', OneHotEncoder(handle_unknown='ignore'))])
        
        self.merge_col = merge_col = ['Commune et arrondissement code', 'Département code']
        self.merge_transformer = FunctionTransformer(self.merge_naive, validate=False)
        
        self.drop_cols = ['Name', 'Coordonnée X', 'Coordonnée Y', 'Commune code', 'City_name',
                          'Commune et arrondissement code', 'Commune et arrondissement nom',
                          'Département nom', 'Académie nom', 'Région nom', 'Région 2016 nom',
                          'Longitude', 'Latitude', 'Position']
        pass

    def fit(self, X_df, y_array):
        X_encoded = X_df
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numeric_transformer, self.num_cols),
                ('cat', self.categorical_transformer, self.cat_cols),
                ('students', make_pipeline(self.students_transformer, SimpleImputer(strategy='mean'), StandardScaler()), self.students_col),
                ('merge', make_pipeline(self.merge_transformer, SimpleImputer(strategy='mean')), self.merge_col),
                ('drop cols', 'drop', self.drop_cols),
                ], remainder='passthrough') # remainder='drop' or 'passthrough'

        self.preprocessor.fit(X_encoded, y_array)
        pass

    def transform(self, X_df):
        X_encoded = X_df
        X_array = self.preprocessor.transform(X_encoded)
        return X_array
    
    @staticmethod
    def process_students(X):
        """Create new features linked to the pupils"""
        # average class size
        X['average_class_size'] = X['Nb élèves'] / X['Nb divisions']
        # percentage of pupils in the general stream
        X['percent_general_stream'] = X['Nb 6èmes 5èmes 4èmes et 3èmes générales'] / X['Nb élèves']
        # percentage of pupils in an european or international section
        X['percent_euro_int_section'] = X['Nb 6èmes 5èmes 4èmes et 3èmes générales sections européennes et internationales'] / X['Nb élèves']
        # percentage of pupils doing Latin or Greek
        sum_global_5_to_3 = X['Nb 5èmes'] + X['Nb 4èmes générales'] + X['Nb 3èmes générales']
        X['percent_latin_greek'] = X['Nb 5èmes 4èmes et 3èmes générales Latin ou Grec'] / sum_global_5_to_3
        # percentage of pupils that are in a SEGPA class
        X['percent_segpa'] = X['Nb SEGPA'] / X['Nb élèves']
        
        return np.c_[X['average_class_size'].values,
                     X['percent_general_stream'].values,
                     X['percent_euro_int_section'].values,
                     X['percent_latin_greek'].values,
                     X['percent_segpa'].values]
        
        
    def merge_naive(self, X):
        # merge the two databases at the city level
        df = pd.merge(X, self.cities_data,
                      left_on='Commune et arrondissement code', right_on='insee_code', how='left')

        # fill na by taking the average value at the departement level
        for col in self.keep_col_cities:
            if self.cities_data[col].isna().sum() > 0:
                df[col] = df[['Département code', col]].groupby('Département code').transform(lambda x: x.fillna(x.mean()))
    
        return df[self.keep_col_cities]
        