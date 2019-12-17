"""This module implements the pipeline steps needed to classify partner choices
in the OpenML Speed Dating challenge."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RangeTransformer(BaseEstimator, TransformerMixin):
    '''
    A custom transformer for ranges.

    Parameters
    ----------
    range_features : list[str]
        This specifies the column names with the ranges.

    Attributes
    ----------
    range_features : list[str]
        Here we store the columns with range features.
    suffix : this determines how we will rename the transformed features.
    '''

    def __init__(self, range_features=[], suffix='_range/mean'):
        assert isinstance(range_features, list)
        self.range_features = range_features
        self.suffix = suffix

    def fit(self, X, y=None):
        '''Nothing to do here
        '''
        return self

    def transform(self, X, y=None):
        '''apply the transformation
        '''
        range_data = pd.DataFrame()
        for col in self.range_features:
            range_data[str(col) + self.suffix] = X[col].apply(
                lambda x: self._encode_ranges(x)
            ).astype(float)
        return range_data

    @staticmethod
    def _encode_ranges(range_str):
        splits = range_str[1:-1].split('-')
        range_max = float(splits[-1])
        range_min = float('-'.join(splits[:-1]))
        return sum([range_min, range_max]) / 2.0


class FloatTransformer(BaseEstimator, TransformerMixin):
    '''
    A custom transformer for floats encoded as strings.

    Parameters
    ----------
    float_features : list[str]
        This specifies the column names with the floats that are encoded as
        strings.

    Attributes
    ----------
    float_features : list[str]
        Here we store the columns with float features.
    suffix : this determines how we will rename the transformed features.
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
        '''
        float_data = pd.DataFrame()
        for col in self.float_features:
            float_data[str(col) + self.suffix] = X[col].apply(
                lambda x: float(x)
                if x != '?' else np.NaN
            ).astype(float)
        return float_data


class PandasPicker(BaseEstimator, TransformerMixin):
    '''
    A convenience class to use pandas dataframes with a pipeline.

    Parameters
    ----------
    features : list[str]
        This specifies the column names that we want to use.

    Attributes
    ----------
    features : list[str]
        Here we store the column names that we use.
    suffix : this determines how we will rename the features.
        Empty string by default.
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
        '''
        new_data = pd.DataFrame()
        for col in self.features:
            new_data[str(col) + self.suffix] = X[col]
        return new_data


class PandasPicker2(PandasPicker):
    '''
    working around this issue:
    https://github.com/openml/OpenML/issues/340

    Found a second occurence of component...
    '''
