# product-search
Home Depot Kaggle contest

[Top 15%](https://www.kaggle.com/qw12qw)

exp-data-analysis.ipynb contains exploratory data analysis relevant for this problem.

cosine_similarity.py contains a transformer object that computes the cosine similarity between each pair of "documents" (eg, search term and product title). It takes two pandas-dataframe column names and a model (eg, TfidfVectorizer instance object) as inputs and it returrns a numpy array n x 1 (with n the number of rows of the original dataframe) with the cosine similarity for each pair of documents. It is compatible with scikit-learn FeatureUniopn and GridSearsch. 
