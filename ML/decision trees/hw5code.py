import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
    feature_vector = np.array(feature_vector)
    
    if len(np.unique(feature_vector)) == 1:
        return [], [], None, None
    
    target_vector = np.array(target_vector)
    target_vector = target_vector[np.argsort(feature_vector)]
    
    feature_vector = np.sort(feature_vector)
    thresholds = (np.unique(feature_vector)[1:] + np.unique(feature_vector)[:-1]) / 2
    
    R = len(target_vector)
    R_l = np.unique(feature_vector, return_index = True)[1][1:]
    R_r = R - R_l
    
    n_1_l = np.cumsum(target_vector)[np.unique(feature_vector, return_index = True)[1][1:] - 1]
    n_1_r = np.cumsum(target_vector[::-1])[len(target_vector) - np.unique(feature_vector, return_index = True)[1][1:] - 1]
    
    H_l = 1 - (n_1_l / R_l) ** 2 - (1 - (n_1_l / R_l)) ** 2
    H_r = 1 - (n_1_r / R_r) ** 2 - (1 - (n_1_r / R_r)) ** 2 
    
    ginis = - R_l / R * H_l - R_r / R * H_r
    
    threshold_best, gini_best = thresholds[np.argmax(ginis)], np.max(ginis)

    return thresholds, ginis, threshold_best, gini_best

class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth):
        sub_X = np.array(sub_X)
        sub_y = np.array(sub_y)

        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])

                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if len(np.unique(feature_vector)) == 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None or len(sub_y[split]) < self._min_samples_leaf or len(sub_y[~split]) < self._min_samples_leaf:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature_split = node["feature_split"]

        if self._feature_types[feature_split] == "real":
            if x[feature_split] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

        elif self._feature_types[feature_split] == "categorical":
            if x[feature_split] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

        else:
            raise ValueError("Unknown feature type")

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, depth=0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class LinearRegressionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=1, n_quantiles=10):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._n_quantiles = n_quantiles

    def _find_best_split(self, X, y):
        n_samples, n_features = X.shape
        best_feature, best_threshold, best_loss = None, None, float('inf')

        for feature in range(n_features):
            feature_type = self._feature_types[feature]
            feature_vector = X[:, feature]

            if feature_type == "real":
                thresholds = np.percentile(feature_vector, np.linspace(0, 100, self._n_quantiles + 1)[1:-1])
            elif feature_type == "categorical":
                thresholds = np.unique(feature_vector)
            else:
                raise ValueError("Unknown feature type")

            for threshold in thresholds:
                left_mask = feature_vector <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < self._min_samples_leaf or np.sum(right_mask) < self._min_samples_leaf:
                    continue

                left_model = LinearRegression().fit(X[left_mask], y[left_mask])
                right_model = LinearRegression().fit(X[right_mask], y[right_mask])

                left_loss = mean_squared_error(y[left_mask], left_model.predict(X[left_mask]))
                right_loss = mean_squared_error(y[right_mask], right_model.predict(X[right_mask]))

                weighted_loss = (np.sum(left_mask) / n_samples) * left_loss + (np.sum(right_mask) / n_samples) * right_loss

                if weighted_loss < best_loss:
                    best_loss = weighted_loss
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_loss

    def _fit_node(self, X, y, node, depth):
        if len(y) < self._min_samples_split or (self._max_depth is not None and depth >= self._max_depth):
            node["type"] = "terminal"
            node["model"] = LinearRegression().fit(X, y)
            return

        best_feature, best_threshold, best_loss = self._find_best_split(X, y)

        if best_feature is None:
            node["type"] = "terminal"
            node["model"] = LinearRegression().fit(X, y)
            return

        node["type"] = "nonterminal"
        node["feature"] = best_feature
        node["threshold"] = best_threshold
        node["left_child"] = {}
        node["right_child"] = {}

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        self._fit_node(X[left_mask], y[left_mask], node["left_child"], depth + 1)
        self._fit_node(X[right_mask], y[right_mask], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["model"].predict(x.reshape(1, -1))[0]

        if x[node["feature"]] <= node["threshold"]:
            return self._predict_node(x, node["left_child"])
        else:
            return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, 0)

    def predict(self, X):
        return np.array([self._predict_node(x, self._tree) for x in X])

