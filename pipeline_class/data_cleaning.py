
import numpy  as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DataCleaning(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = self.data_cleaning(X)

        return X

    def data_cleaning(self, data ):
        """
        Performn data cleaning in DataFrame.
        """
        # Convert Invoice Date
        data.invoice_date = pd.to_datetime(data.invoice_date)

        # Stock_code
        stock_code = ['POST', 'D', 'DOT', 'M', 'S', 'AMAZONFEE', 'm', 'DCGSSBOY', 'DCGSSGIRL', 'PADS', 'B', 'CRUK']
        data = data.query('stock_code not in @stock_code')

        # Unit_price
        data = data.query('unit_price > 0.01')
        data = data.dropna()

        return data