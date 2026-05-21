import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, '.')
from fusion.tensor import tensor
from fusion.linear_model import LinearRegression

# Загрузка данных
data = fetch_california_housing()
X_np, y_np = data.data, data.target.reshape(-1, 1)
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_np, y_np, test_size=0.2, random_state=42
)

# Нормализация признаков (иначе градиентный спуск расходится)
scaler = StandardScaler()
X_train_np = scaler.fit_transform(X_train_np)
X_test_np = scaler.transform(X_test_np)

# Целевую переменную тоже желательно нормализовать для устойчивости
y_scaler = StandardScaler()
y_train_np = y_scaler.fit_transform(y_train_np)
y_test_np = y_scaler.transform(y_test_np)

# Преобразование в тензоры Fusion
X_train = tensor(X_train_np.flatten().tolist(), list(X_train_np.shape))
y_train = tensor(y_train_np.flatten().tolist(), list(y_train_np.shape))
X_test  = tensor(X_test_np.flatten().tolist(), list(X_test_np.shape))
y_test  = tensor(y_test_np.flatten().tolist(), list(y_test_np.shape))

# Обучение линейной регрессии (уменьшенный lr)
model = LinearRegression(lr=0.01, epochs=1000)
model.fit(X_train, y_train)

# Предсказания
pred_train = model.predict(X_train)
pred_test  = model.predict(X_test)

# Метрика MSE на тензорах
def fusion_mse(y_true, y_pred):
    diff = y_pred - y_true
    return (diff * diff).mean()

train_mse = fusion_mse(y_train, pred_train)
test_mse  = fusion_mse(y_test, pred_test)

print(f"Train MSE: {train_mse:.4f}")
print(f"Test  MSE: {test_mse:.4f}")
print(f"Train RMSE: {np.sqrt(train_mse):.4f}")
print(f"Test  RMSE: {np.sqrt(test_mse):.4f}")