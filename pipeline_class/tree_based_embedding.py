import umap
import numpy          as np
import pandas         as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.base     import BaseEstimator, TransformerMixin

class TreeBasedEmbedding(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = self.tree_based_embedding(X)

        return X

    def tree_based_embedding(self, data ):
        """
        Create tree-based embedding using random forest regression and perform dimensionality reduction with UMAP.
        """

        cols_selected = ['recency', 'frequency', 'number_products', 'nunique_products', 'return_rate', 
                      'avg_order_value', 'avg_basket_size', 'lifetime']
        #-------------------------------------------------------------------------------------------------
        # Create Tree Based Embedding
        # Split X and y
        X_train = data[cols_selected]
        y_train = data.gross_revenue

        # Intancing
        rf = RandomForestRegressor( n_estimators=100, random_state=0 )

        # Training
        rf.fit(X_train, y_train)

        # Leafs ( Tree Based Embedding)
        tree_based_dataframe = pd.DataFrame( rf.apply( X_train ) )

        #-------------------------------------------------------------------------------------------------
        # Dimensionality reduction with UMAP

        reducer = umap.UMAP( random_state=0, n_neighbors=60, min_dist=0.01  )
        embedding = reducer.fit_transform( tree_based_dataframe )

        # embedding
        embedding_dataframe = pd.DataFrame()
        embedding_dataframe['embedding_x'] = embedding[:, 0]
        embedding_dataframe['embedding_y'] = embedding[:, 1]

        return embedding_dataframe