import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 📌 Загрузка и обработка
df = pd.read_csv('taxi_trip_pricing.csv')
df.dropna(inplace=True)
for col in ['Time_of_Day', 'Day_of_Week', 'Traffic_Conditions', 'Weather']:
    df[col] = df[col].astype('category').cat.codes
X = df.drop('Trip_Price', axis=1).values
y = df['Trip_Price'].values
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]


# 📌 Метрика
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


# 📌 KNN
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def knn_regressor(X_train, y_train, X_test, k=3):
    predictions = []
    for x in X_test:
        distances = [euclidean_distance(x, x_train) for x_train in X_train]
        k_indices = np.argsort(distances)[:k]
        k_values = y_train[k_indices]
        predictions.append(np.mean(k_values))
    return np.array(predictions)


y_pred_knn = knn_regressor(X_train, y_train, X_test, k=3)


# 📌 Decision Tree
class DecisionTreeRegressor:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth

    def mse(self, y):
        return np.mean((y - np.mean(y)) ** 2)

    def best_split(self, X, y):
        best_score = float('inf')
        best_split = None
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left = y[X[:, feature] <= t]
                right = y[X[:, feature] > t]
                if len(left) == 0 or len(right) == 0:
                    continue
                score = len(left)/len(y)*self.mse(left) + len(right)/len(y)*self.mse(right)
                if score < best_score:
                    best_score = score
                    best_split = (feature, t)
        return best_split

    def fit(self, X, y, depth=0):
        if depth == self.max_depth or len(set(y)) == 1:
            return np.mean(y)
        split = self.best_split(X, y)
        if split is None:
            return np.mean(y)
        feature, threshold = split
        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold
        left = self.fit(X[left_indices], y[left_indices], depth + 1)
        right = self.fit(X[right_indices], y[right_indices], depth + 1)
        return (feature, threshold, left, right)

    def predict_single(self, x, tree):
        if not isinstance(tree, tuple):
            return tree
        feature, threshold, left, right = tree
        return self.predict_single(x, left) if x[feature] <= threshold else self.predict_single(x, right)

    def predict(self, X):
        return np.array([self.predict_single(x, self.tree) for x in X])

    def train(self, X, y):
        self.tree = self.fit(X, y)


tree = DecisionTreeRegressor(max_depth=3)
tree.train(X_train, y_train)
y_pred_tree = tree.predict(X_test)

# 📌 Ансамбль
y_ensemble = (y_pred_knn + y_pred_tree) / 2

# 📌 Метрики
mae_knn = mean_absolute_error(y_test, y_pred_knn)
mae_tree = mean_absolute_error(y_test, y_pred_tree)
mae_ensemble = mean_absolute_error(y_test, y_ensemble)

print("📊 MAE:")
print("KNN:     ", mae_knn)
print("Tree:    ", mae_tree)
print("Ensemble:", mae_ensemble)

# 📈 Графики

# 1. Распределение целевой переменной
plt.figure(figsize=(8,4))
plt.hist(y, bins=30, color='skyblue', edgecolor='black')
plt.title("Распределение цены поездки")
plt.xlabel("Цена поездки")
plt.ylabel("Количество")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. График: Реальные vs Предсказанные значения
plt.figure(figsize=(10,5))
plt.plot(y_test[:50], label='Истинные значения', marker='o')
plt.plot(y_pred_knn[:50], label='KNN', linestyle='--')
plt.plot(y_pred_tree[:50], label='Дерево решений', linestyle='--')
plt.plot(y_ensemble[:50], label='Ансамбль', linestyle='--')
plt.title("Сравнение предсказаний (первые 50 тестовых точек)")
plt.ylabel("Цена поездки")
plt.xlabel("Номер примера")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. MAE диаграмма
plt.bar(['KNN', 'Tree', 'Ensemble'], [mae_knn, mae_tree, mae_ensemble], color=['orange', 'lightgreen', 'skyblue'])
plt.title("Средняя абсолютная ошибка моделей")
plt.ylabel("MAE")
plt.tight_layout()
plt.show()

# 🔍 Рассеяние KNN
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_knn, alpha=0.6, color='orange', label='KNN')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Идеал')
plt.xlabel("Истинная цена")
plt.ylabel("Предсказанная цена")
plt.title("Диаграмма рассеяния: KNN")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 🌳 Рассеяние дерева решений
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_tree, alpha=0.6, color='green', label='Дерево решений')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Идеал')
plt.xlabel("Истинная цена")
plt.ylabel("Предсказанная цена")
plt.title("Диаграмма рассеяния: Дерево решений")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 🧠 Рассеяние ансамбля
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_ensemble, alpha=0.6, color='blueviolet', label='Ансамбль')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Идеал')
plt.xlabel("Истинная цена")
plt.ylabel("Предсказанная цена")
plt.title("Диаграмма рассеяния: Ансамбль")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
