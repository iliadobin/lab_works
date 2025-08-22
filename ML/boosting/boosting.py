from __future__ import annotations

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor

from typing import Optional


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])

class Boosting:
    def __init__(
        self,
        base_model_params: dict = None,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        subsample: float | int = 1.0,
        bagging_temperature: float | int = 1.0,
        bootstrap_type: str | None = 'Bernoulli',
        goss: bool | None = False,
        goss_k: float | int = 0.2,
        goss_subsample: float | int = 0.3,
        early_stopping_rounds: int = None,
        plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators
        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float | int = subsample
        self.bagging_temperature: float | int = bagging_temperature
        self.bootstrap_type: str | None = bootstrap_type

        self.goss: bool | None = goss
        self.goss_k: float | int = goss_k
        self.goss_subsample: float | int = goss_subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def _bernoulli_bootstrap(self, y, size):
        return np.random.rand(size) < self.subsample

    def _bayesian_bootstrap(self, size):
        X = np.random.rand(size)
        return (-np.log(X)) ** self.bagging_temperature

    def _goss_sampling(self, X, y, gradients):
        n_samples = len(gradients)
        top_k = int(self.goss_k * n_samples)
        remaining_k = int(self.goss_subsample * (n_samples - top_k))

        # Индексы объектов с большими градиентами
        top_indices = np.argsort(-np.abs(gradients))[:top_k]
        small_indices = np.argsort(-np.abs(gradients))[top_k:]

        # Сэмплирование объектов с маленькими градиентами
        sampled_indices = np.random.choice(
            small_indices, size=remaining_k, replace=False
        )
        selected_indices = np.concatenate([top_indices, sampled_indices])

        # Масштабирование весов
        weights = np.ones(n_samples)
        weights[sampled_indices] *= n_samples / remaining_k

        return X[selected_indices], y[selected_indices], weights[selected_indices]

    def partial_fit(self, X, y, predictions):
        new_model = self.base_model_class(**self.base_model_params)
        size = y.shape[0]

        gradients = -self.loss_derivative(y, predictions)

        if self.goss:
            X_train, y_train, weights = self._goss_sampling(X, y, gradients)
        elif self.bootstrap_type == 'Bernoulli':
            mask = self._bernoulli_bootstrap(y, size)
            X_train = X[mask, :]
            y_train = y[mask]
            weights = None
        elif self.bootstrap_type == 'Bayesian':
            weights = self._bayesian_bootstrap(size)
            X_train = X
            y_train = y
        else:
            raise ValueError("Invalid bootstrap_type. Must be 'Bernoulli' or 'Bayesian'.")

        y_pred = predictions[:len(y_train)]
        s = -self.loss_derivative(y_train, y_pred)
        new_model = new_model.fit(X_train, s, sample_weight=weights)
        new_gamma = self.find_optimal_gamma(y_train, y_pred, new_model.predict(X_train))
        self.gammas.append(self.learning_rate * new_gamma)
        self.models.append(new_model)

    def fit(self, X_train, y_train, X_val=None, y_val=None, plot=False):
        """
        :param X_train: features array (train set)
        :param y_train: targets array (train set)
        :param X_val: features array (eval set)
        :param y_val: targets array (eval set)
        :param plot: bool 
        """
        train_predictions = np.zeros(y_train.shape[0])
        if X_val is not None:
            valid_predictions = np.zeros(y_val.shape[0])
        loss_train = []
        loss_val = []

        for i in range(self.n_estimators):
            self.partial_fit(X_train, y_train, train_predictions)
            train_predictions = np.round(self.predict_proba(X_train)[:, 1])
            loss_train.append(self.loss_fn(y_train, train_predictions))
            if X_val is not None:
                valid_predictions = np.round(self.predict_proba(X_val)[:, 1])
                loss_val.append(self.loss_fn(y_val, valid_predictions))

            if self.early_stopping_rounds is not None:
                if i > self.early_stopping_rounds:
                    break
                if X_val is not None and len(loss_val)>2:
                    if loss_val[-1] > loss_val[-2]:
                        break
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(loss_train)), loss_train, label='train_loss')
            if X_val is not None:
                plt.plot(range(len(loss_val)), loss_val, label='valid_loss')
            plt.title("Boosting results")
            plt.ylabel("Loss")
            plt.xlabel("n_estimator")
            plt.legend()
            plt.show()

    def predict_proba(self, X):
        predictions = np.zeros(X.shape[0])
        for gamma, model in zip(self.gammas, self.models):
            predictions += model.predict(X) * self.learning_rate * gamma
        predictions = self.sigmoid(predictions)
        probs = np.zeros([X.shape[0], 2])
        probs[:, 0], probs[:, 1] = 1 - predictions, predictions
        return probs

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def score(self, X, y):
        return score(self, X, y)

    def plot_history(self, X, y):
        """
        Визуализирует историю значений функции потерь и ROC-AUC на переданном наборе данных.

        :param X: features array (any set)
        :param y: targets array (any set)
        """
        if not self.models:
            raise Exception("Модели еще не обучены. Сначала вызовите fit.")

        predictions = np.zeros(X.shape[0])
        loss_history = []
        roc_auc_history = []

        for gamma, model in zip(self.gammas, self.models):
            predictions += model.predict(X) * self.learning_rate * gamma
            loss = self.loss_fn(y, predictions)
            roc_auc = roc_auc_score(y == 1, self.sigmoid(predictions))

            loss_history.append(loss)
            roc_auc_history.append(roc_auc)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(loss_history) + 1), loss_history, label="Loss", marker="o")
        plt.title("Loss History")
        plt.xlabel("Number of Models")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(roc_auc_history) + 1), roc_auc_history, label="ROC-AUC", marker="o", color="orange")
        plt.title("ROC-AUC History")
        plt.xlabel("Number of Models")
        plt.ylabel("ROC-AUC")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    @property
    def feature_importances_(self):
        """
        Возвращает важности признаков как среднее значений feature_importances_ 
        у всех базовых моделей, нормированных до суммы 1.
        """
        if not self.models:
            raise Exception("Модели еще не обучены. Вызовите fit)")

        total_importances = np.zeros_like(self.models[0].feature_importances_)

        for model in self.models:
            total_importances += model.feature_importances_

        total_importances /= total_importances.sum()

        return total_importances


