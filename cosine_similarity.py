import pandas as pd 
import numpy as np 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import linear_kernel

# I might apply dimensional reduction (LatentSematicAnalysis) 
# to tf-idf vectors, before computing cosine similarity, so as
# to reduce sparsity

class CosineSimilarity(BaseEstimator, TransformerMixin):
    '''
    Transformer that computes cosine similarity between the counting or 
    tf-idf (model) of two features given by col_names. 
    INPUTS: 
    model: tf-idf or couning model
    col_names: a 2-tupple of column names. Only the first column is used to 
                fit the model. Both are transformed. 
    df: dataframe 
    '''

    def __init__(self, model, col_names):
        self.model = model
        self.col_names = col_names

    def fit(self, df, y=None, **kwargs):
        A = self.select(df, self.col_names[0])
        self.model.fit(A, **kwargs)
        return self

    def select(self, df, col_name):
        return df[col_name].values

    def transform(self, df):
        
        A = self.select(df, self.col_names[0])
        A = self.model.transform(A)
        B = self.select(df, self.col_names[1])  
        B = self.model.transform(B)
        nrows = A.shape[0] 
        # Given that tf-idf vectorizer normalized vectors, we can use 
        # a linear_kernel (scalar product) to cumpute the cosine
        X = np.array([linear_kernel(A[i], B[i]) for i in range(nrows)])
        X = X.reshape((nrows, 1))

        return X



