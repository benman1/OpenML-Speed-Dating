#!/usr/bin/env python

"""Tests for `openml_speed_dating_pipeline_steps` package."""


import unittest

from sklearn import datasets
import pandas as pd
from pandas.api.types import is_numeric_dtype
from openml_speed_dating_pipeline_steps import (
    openml_speed_dating_pipeline_steps
    as pipeline_steps
)


class TestOpenml_speed_dating_pipeline_steps(unittest.TestCase):
    """Tests for `openml_speed_dating_pipeline_steps` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        iris = datasets.load_iris()
        self.data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        self.range_col = iris.feature_names[0] + 'range'
        self.range_orig = iris.feature_names[0]
        self.data[self.range_col] = self.data[iris.feature_names[0]].apply(
            lambda x: '[{}-{}]'.format(x, x+1)
        )
        self.numeric_difference = pipeline_steps.NumericDifferenceTransformer()
        self.range_transformer = pipeline_steps.RangeTransformer()

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_numeric_difference_columns(self):
        """Test that numeric differences returns the
        right number of columns."""
        assert(len(self.numeric_difference.transform(self.data).columns) == 6)

    def test_001_numeric_difference_coltypes(self):
        transformed = self.numeric_difference.transform(self.data)
        for col in transformed.columns:
            assert is_numeric_dtype(transformed[col])

    def test_002_range_columns(self):
        """Test that numeric differences returns the
        right number of columns."""
        assert(len(
            self.range_transformer.transform(
                self.data[self.range_col]
            ).columns
        ) == 1)

    def test_003_range_coltypes(self):
        transformed = self.range_transformer.transform(
            self.data[self.range_col]
        )
        for col in transformed.columns:
            assert is_numeric_dtype(transformed[col])

    def test_004_range_content(self):
        transformed = self.range_transformer.transform(
            self.data[self.range_col]
        )
        col = list(transformed.columns)[0]
        assert ((
            transformed[col] -
            (self.data[self.range_orig] + 0.5)
        ) < 0.01).all()
