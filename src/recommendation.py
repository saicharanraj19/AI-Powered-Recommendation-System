import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def recommend_content_based(data):
    # Assuming 'item_description' is a column with text descriptions of items
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['item_description'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

def recommend_collaborative(data):
    # Create a user-item matrix
    user_item_matrix = pd.pivot_table(data, index='userId', columns='itemId', values='rating')
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(user_item_matrix.fillna(0))

    # Apply SVD
    svd = TruncatedSVD(n_components=50)
    matrix_reduced = svd.fit_transform(scaled_matrix)
    return matrix_reduced
