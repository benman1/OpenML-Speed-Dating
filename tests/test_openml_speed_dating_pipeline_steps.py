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
        self.numeric_difference = pipeline_steps.NumericDifferenceTransformer()

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_numeric_difference_columns(self):
        """Test that numeric differences returns the
        right number of columns."""
        assert(len(self.numeric_difference.transform(self.data).columns) == 6)

    def test_001_numeric_difference_colnames(self):
        transformed = self.numeric_difference.transform(self.data)
        for col in transformed.columns:
            assert is_numeric_dtype(transformed[col])
