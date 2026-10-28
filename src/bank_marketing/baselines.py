from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from .evaluate import metrics


def fit_baselines(train_data, seed=42):
    models = {'dummy_baseline': DummyClassifier(strategy='most_frequent'),
              'logistic_regression_baseline': LogisticRegression(class_weight='balanced', max_iter=2000, random_state=seed)}
    for model in models.values():
        model.fit(train_data.x, train_data.y)
    return models


def evaluate_baselines(models, test_data):
    return {name: metrics(test_data.y, model.predict_proba(test_data.x)[:, 1]) for name, model in models.items()}

