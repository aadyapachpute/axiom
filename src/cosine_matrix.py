import numpy as np

def pairwise_cosine_similarity(X):
    magnitudes = np.sqrt(np.sum(X ** 2, axis=1))


    dot_products = X @ X.T

    similarity_matrix = dot_products / (
    magnitudes[:, None] * magnitudes[None, :]
)

    return similarity_matrix


X = np.array([
    [1, 2, 3],
    [2, 4, 6],
    [1, 0, 1]
])


result = pairwise_cosine_similarity(X)
print(result)
