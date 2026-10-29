# Decision Tree from the original research, with scikit-learn's default settings.
from sklearn.tree import DecisionTreeRegressor

NAME = 'Decision Tree'


def build_model(seed=None):
    return DecisionTreeRegressor(random_state=seed)
