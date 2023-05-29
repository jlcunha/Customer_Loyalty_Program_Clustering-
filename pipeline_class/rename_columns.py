
import inflection
import numpy  as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RenameDataframeColumns(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = self.rename_columns(X)

        return X

    def rename_columns(self, data):
        # Copy DataFrame
        new_cols = ['invoice_no', 'stock_code', 'description', 'quantity', 'invoice_date', 'unit_price', 'customer_id', 'country']

        # Apply new columns names to DataFrame
        data.columns = new_cols
        
        return data