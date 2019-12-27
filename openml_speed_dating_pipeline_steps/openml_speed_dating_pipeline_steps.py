"""This module implements the pipeline steps needed to classify partner choices
in the OpenML Speed Dating challenge."""
from functools import lru_cache
import operator
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import category_encoders.utils as util


class RangeTransformer(BaseEstimator, TransformerMixin):
    '''
    A custom transformer for ranges.

    Parameters
    ----------
    range_features : list[str] or None
        This specifies the column names with the ranges. If None,
        all features will be encoded. This is important so this
        transformer will work with sklearn's ColumnTransformer.
    suffix : this determines how we will rename the transformed features.

    Attributes
    ----------
    range_features : list[str]
        Here we store the columns with range features.
    '''
    def __init__(self, range_features=None, suffix='_range/mean', n_jobs=-1):
        assert isinstance(range_features, list) or range_features is None
        self.range_features = range_features
        self.suffix = suffix
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        '''Nothing to do here
        '''
        return self

    def transform(self, X, y=None):
        '''apply the transformation

        Parameters:
        -----------
        X : array-like; either numpy array or pandas dataframe.
        '''
        X = util.convert_input(X)
        if self.range_features is None:
            self.range_features = list(X.columns)

        range_data = pd.DataFrame(index=X.index)
        for col in self.range_features:
            range_data[str(col) + self.suffix] = pd.to_numeric(
                self._vectorize(X[col])
            )
        self.feature_names = list(range_data.columns)
        return range_data

    def _vectorize(self, s):
        return Parallel(n_jobs=self.n_jobs)(
            delayed(self._encode_range)(x) for x in s
        )

    @staticmethod
    @lru_cache(maxsize=32)
    def _encode_range(range_str):
        splits = range_str[1:-1].split('-')
        range_max = float(splits[-1])
        range_min = float('-'.join(splits[:-1]))
        return sum([range_min, range_max]) / 2.0

    def get_feature_names(self):
        '''Array mapping from feature integer indices to feature name
        '''
        return self.feature_names


class NumericDifferenceTransformer(BaseEstimator, TransformerMixin):
    '''
    A custom transformer that calculates differences between
    numeric features.

    Parameters
    ----------
    features : list[str] or None
        This specifies the column names with the numerical features. If None,
        all features will be encoded. This is important so this
        transformer will work with sklearn's ColumnTransformer.
    suffix : this determines how we will rename the transformed features.
    op : this is the operation to calculate between the two columns.
        This is minus (operator.sub) by default.

    Attributes
    ----------
    features : list[str]
        Here we store the columns with numerical features.


    Example
    -------
    >>> from sklearn import datasets
    >>> import pandas as pd
    >>> iris = datasets.load_iris()
    >>> data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    >> numeric_difference = pipeline_steps.NumericDifferenceTransformer()
    >>> numeric_difference.transform(data).columns
    Index(['sepal length (cm)_sepal width (cm)_numdist',
           'sepal length (cm)_petal length (cm)_numdist',
           'sepal length (cm)_petal width (cm)_numdist',
           'sepal width (cm)_petal length (cm)_numdist',
           'sepal width (cm)_petal width (cm)_numdist',
           'petal length (cm)_petal width (cm)_numdist'],
          dtype='object')
    '''
    def __init__(self, features=None,
                 suffix='_numdist', op=operator.sub, n_jobs=-1
                 ):
        assert isinstance(features, list) or features is None
        self.features = features
        self.suffix = suffix
        self.op = op
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        '''Nothing to do here
        '''
        X = util.convert_input(X)
        if self.features is None:
            self.features = list(
                X.select_dtypes(include='number').columns
            )
        return self

    def _col_name(self, col1, col2):
        return str(col1) + '_' + str(col2) + self.suffix

    def _feature_pairs(self):
        feature_pairs = []
        for i, col1 in enumerate(self.features[:-1]):
            for col2 in self.features[i+1:]:
                feature_pairs.append((col1, col2))
        return feature_pairs

    def transform(self, X, y=None):
        '''apply the transformation

        Parameters:
        -----------
        X : array-like; either numpy array or pandas dataframe.
        '''
        X = util.convert_input(X)

        feature_pairs = self._feature_pairs()
        columns = Parallel(n_jobs=self.n_jobs)(
            delayed(self._col_name)(col1, col2)
            for col1, col2 in feature_pairs
        )
        data_cols = Parallel(n_jobs=self.n_jobs)(
            delayed(self.op)(X[col1], X[col2])
            for col1, col2 in feature_pairs
        )
        data = pd.concat(data_cols, axis=1)
        data.rename(
            columns={i: col for i, col in enumerate(columns)},
            inplace=True, copy=False
        )
        data.index = X.index
        return data

    def get_feature_names(self):
        '''Array mapping from feature integer indices to feature name
        '''
        return self.feature_names


class FloatTransformer(BaseEstimator, TransformerMixin):
    '''
    A custom transformer for floats encoded as strings.
    NOTE: I consider this tranformer obsolete, since
    I am using the OpenML version of the dataset.

    Parameters
    ----------
    float_features : list[str] or None
        This specifies the column names with the floats that are encoded as
        strings.
    suffix : this determines how we will rename the transformed features.

    Attributes
    ----------
    float_features : list[str] or None
        Here we store the columns with float features.
    '''

    def __init__(self, float_features=[], suffix='_asfloat'):
        assert isinstance(float_features, list)
        self.float_features = float_features
        self.suffix = suffix

    def fit(self, X, y=None):
        '''Nothing to do here
        '''
        return self

    def transform(self, X, y=None):
        '''apply the transformation

        Parameters:
        -----------
        X : array-like; either numpy array or pandas dataframe.
        '''
        X = util.convert_input(X)
        if self.float_features is None:
            self.float_features = list(X.columns)

        float_data = pd.DataFrame()
        for col in self.float_features:
            float_data[str(col) + self.suffix] = X[col].apply(
                lambda x: float(x)
                if x != '?' else np.NaN
            ).astype(float)
        self.feature_names = list(float_data.columns)
        return float_data

    def get_feature_names(self):
        '''Array mapping from feature integer indices to feature name
        '''
        return self.feature_names


class PandasPicker(BaseEstimator, TransformerMixin):
    '''
    A convenience class to use pandas dataframes with a pipeline.

    Parameters
    ----------
    features : list[str]
        This specifies the column names that we want to use.
    suffix : this determines how we will rename the features.
        Empty string by default.

    Attributes
    ----------
    features : list[str]
        Here we store the column names that we use.
    '''

    def __init__(self, features=[], suffix=''):
        assert isinstance(features, list)
        self.features = features
        self.suffix = suffix

    def fit(self, X, y=None):
        '''Nothing to do here
        '''
        return self

    def transform(self, X, y=None):
        '''apply the transformation

        Parameters:
        -----------
        X : array-like; either numpy array or pandas dataframe.
        '''
        X = util.convert_input(X)
        if self.features is None:
            self.features = list(X.columns)

        new_data = pd.DataFrame()
        for col in self.features:
            new_data[str(col) + self.suffix] = X[col]
        return new_data

    def get_feature_names(self):
        '''Array mapping from feature integer indices to feature name
        '''
        return self.features


class PandasPicker2(PandasPicker):
    '''
    working around this issue:
    https://github.com/openml/OpenML/issues/340

    Found a second occurence of component...
    '''


class SimpleImputerWithFeatureNames(SimpleImputer):
    '''Thin wrapper around the SimpleImputer that provides get_feature_names()
    '''
    def __init__(self, missing_values=np.nan, strategy="mean",
                 fill_value=None, verbose=0, copy=True):
        super(SimpleImputerWithFeatureNames, self).__init__(
            missing_values, strategy, fill_value, verbose,
            copy, add_indicator=True
        )

    def fit(self, X, y=None):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            self.features = list(X.columns)
        else:
            self.features = list(range(X.shape[1]))
        return super().fit(X, y)

    def get_feature_names(self):
        return [self.features[f] for f in self.indicator_.features_]
