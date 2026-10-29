# Extra Trees from the original research: 3 trees, as in the original code.
from sklearn.ensemble import ExtraTreesRegressor

NAME = 'Extra Trees'


def build_model(seed=None):
    return ExtraTreesRegressor(n_estimators=3, random_state=seed, n_jobs=-1)
