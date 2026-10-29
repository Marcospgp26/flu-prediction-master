# Random Forest from the original research: 23 trees, as in the original code.
from sklearn.ensemble import RandomForestRegressor

NAME = 'Random Forest'


def build_model(seed=None):
    return RandomForestRegressor(n_estimators=23, random_state=seed, n_jobs=-1)
