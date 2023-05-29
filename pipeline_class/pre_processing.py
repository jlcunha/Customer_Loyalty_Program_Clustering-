import numpy          as np
import pandas         as pd
from sklearn.base          import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

class PreProcessing(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = self.pre_processing(X)

        return X

    def pre_processing(self, data):
        """
        Perform pre processing on the input data by dropping 'customer_id' column and applying Min-Max scaling.
        """

        # Drop 'customer_id' column
        data = data.drop(columns='customer_id')

        # Perform Min-Max scaling
        cols = data.columns

        minmax = MinMaxScaler()

        data = pd.DataFrame(minmax.fit_transform(data), columns=cols)

        return data