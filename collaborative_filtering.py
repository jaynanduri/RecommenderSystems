import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm


class Recommend(BaseEstimator, ClassifierMixin):
    """
    This class implements a recommender system using Pearson Correlation as the similarity metric.
    """

    def __init__(self, n_users: int) -> None:
        """
        Initialise similarity matrix and number of users.
        :param n_users: number of users
        """
        self.n_users = n_users
        self.similarities = np.zeros((n_users, n_users))

    def fit(self, X: np.ndarray, y=None) -> None:
        """
        Compute similarity scores for given training data set.
        :param X: User rating data
        :param y:
        :return:
        """
        train_ratings = X[X[:, 0].argsort()].copy()
        train_ratings = np.split(train_ratings[:, 1:3], np.unique(train_ratings[:, 0], return_index=True)[1][1:])

        for user1_id, user1_ratings in enumerate(tqdm(train_ratings)):

            for _, user2_ratings in enumerate(train_ratings[user1_id + 1:]):

                user2_id = user1_id + 1

                overlap1 = user1_ratings[np.isin(user1_ratings[:, 0], user2_ratings[:, 0])]
                # If there are no ratings in common, skip
                if overlap1.shape[0] < 1:
                    continue
                overlap1 = overlap1[overlap1[:, 0].argsort()]
                norm_ratings1 = overlap1[:, 1] - overlap1[:, 1].mean(axis=0, keepdims=True)

                overlap2 = user2_ratings[np.isin(user2_ratings[:, 0], user1_ratings[:, 0])]
                overlap2 = overlap2[overlap2[:, 0].argsort()]
                norm_ratings2 = overlap2[:, 1] - overlap2[:, 1].mean(axis=0, keepdims=True)

                numerator = np.inner(norm_ratings1, norm_ratings2)
                similarity = 0.0
                if numerator != 0.0:

                    denominator = np.sqrt(np.square(norm_ratings1).sum()) * np.sqrt(np.square(norm_ratings1).sum())
                    similarity = np.divide(numerator, denominator)

                # May as well fill in both entries in matrix
                self.similarities[user1_id, user2_id] = similarity
                self.similarities[user2_id, user1_id] = similarity

    def predict(self, X: np.ndarray, data: np.ndarray) -> None:
        """
        A prediction (user, movie) is the weighted average of other users' rating for the movie.
        :param data: Total data before splitting it
        :param X: Test data
        :return:
        """
        total = 0
        squares = 0

        for ratings in tqdm(X):
            user_id = ratings[0][0] - 1
            for rating in ratings:
                movie_id = rating[1]
                true_rating = rating[2]

                other_ratings = data[np.where(data[:, 1] == movie_id)]
                weights = np.array([self.similarities[user_id, id - 1] for id in other_ratings[:, 0]])
                similarities_sum = np.sum(weights)
                if similarities_sum:
                    prediction = np.sum(weights * other_ratings[:, 2]) / similarities_sum
                    total += 1
                    squares += np.square(true_rating - prediction)
        print(f'RMSD over test data: {np.sqrt(squares / total)}')


if __name__ == "__main__":
    data = np.genfromtxt('data/ml-100k/u.data', dtype=np.int32)
    train_data = data[:int(data.shape[0] * 0.8), :]
    test_data = data[int(data.shape[0] - data.shape[0] * 0.2):, :]
    test_ratings = test_data[test_data[:, 0].argsort()].copy()
    test_ratings = np.split(test_ratings[:, 0:3], np.unique(test_ratings[:, 0], return_index=True)[1][1:])
    num_users = max(np.unique(train_data[:, 0]))
    movie_recommend = Recommend(num_users)
    movie_recommend.fit(train_data)
    print(movie_recommend.similarities)
    movie_recommend.predict(test_ratings, data)
