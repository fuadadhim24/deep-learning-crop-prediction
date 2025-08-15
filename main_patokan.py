from warnings import filterwarnings

filterwarnings("ignore")

from datetime import datetime
import typing as tp
from IPython.display import display
from pylab import rcParams
import matplotlib.pyplot as plt
import seaborn as sns

rcParams["figure.figsize"] = 15, 7  # Размер графиков по умолчанию
sns.set(palette="Set2", font_scale=1.2)  # Стиль Seaborn

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_squared_error,      # MSE (среднеквадратичная ошибка)
    mean_absolute_error,     # MAE (средняя абсолютная ошибка)
    mean_absolute_percentage_error,  # MAPE (относительная ошибка)
)

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, Huber

pred_csv_path = "dataset/yield_df.csv"

x = pd.read_csv(pred_csv_path)
x.head()

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.regularizers import l2

# Загрузка и подготовка данных
data = pd.read_csv(pred_csv_path)
maize_data = data[data['Item'] == 'Maize'].dropna().sort_values(['Area', 'Year'])

# Предварительная обработка и удаление пропусков
print(f"Исходный размер датасета: {data.shape}")

# Удаление строк с пропущенными значениями в ключевых столбцах
features = ['avg_temp', 'average_rain_fall_mm_per_year', 'pesticides_tonnes']  
target = 'hg/ha_yield'

# Нормализация данных
scaler_dict = {}
for col in features + [target]:
    scaler = MinMaxScaler()
    maize_data[col] = scaler.fit_transform(maize_data[[col]])
    scaler_dict[col] = scaler
look_back=9

def create_test_sequences_for_year(data, features, target, year, look_back=9):
    X, y, areas = [], [], []
    areas_unique = data['Area'].unique()

    for area in areas_unique:
        region_data = data[data['Area'] == area].sort_values('Year')
        years = region_data['Year'].values

        if year in years:
            year_idx = np.where(years == year)[0][0]
            if year_idx >= look_back:
                seq = region_data.iloc[year_idx - look_back:year_idx]
                target_val = region_data.iloc[year_idx][target]

                if seq[features].isnull().values.any() or pd.isnull(target_val):
                    continue  # пропуск если NaN

                X.append(seq[features].values)
                y.append(target_val)
                areas.append(area)

    return np.array(X), np.array(y), areas

# Вызов:
X_2012, y_2012, area_names_2012 = create_test_sequences_for_year(maize_data, features, target, 2012, look_back=9)

def create_sequences_all(data, features, target, look_back=9):
    X, y = [], []
    areas = data['Area'].unique()
    for area in areas:
        region_data = data[data['Area'] == area].sort_values('Year')
        for i in range(look_back, len(region_data)):
            X.append(region_data[features].iloc[i-look_back:i].values)
            y.append(region_data[target].iloc[i])
    return np.array(X), np.array(y)

X, y = create_sequences_all(maize_data, features, target, look_back=9)

print(f"Размер обучающей выборки: {X.shape}, метки: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 1. SimpleRNN модель
def build_rnn_model():
    model = Sequential([
        SimpleRNN(64, activation='tanh', input_shape=(look_back, len(features))),
        Dropout(0.2),
        Dense(64, activation='relu', kernel_regularizer='l2'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


# 2. LSTM модель
def build_lstm_model():
    model = Sequential([
        LSTM(64, activation='tanh', return_sequences=True, input_shape=(look_back, len(features))),
        Dropout(0.2),
        LSTM(32, activation='tanh'),
        Dense(64, activation='relu', kernel_regularizer='l2'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


# 3. GRU модель
def build_gru_model():
    model = Sequential([
        GRU(64, activation='tanh', return_sequences=True, input_shape=(look_back, len(features))),
        Dropout(0.2),
        GRU(32),
        Dense(64, activation='relu', kernel_regularizer='l2'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model



from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

def evaluate_model(model, name, X_2012, y_2012, scaler, target, area_names):
    y_pred_scaled = model.predict(X_2012)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_2012.reshape(-1, 1))

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n== Метрики для {name} на 2012 год ==")
    print(f"MAE: {mae:.2f} hg/ha")
    print(f"MSE: {mse:.2f} hg/ha")
    print(f"R² : {r2:.4f}")
    return y_pred, y_true

try:
    models = {
        "SimpleRNN": build_rnn_model(),
        "LSTM": build_lstm_model(),
        "GRU": build_gru_model()
    }

    predictions = {}

    for name, model in models.items():
        print(f"\nОбучение {name}...")
        model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1)

        y_pred, y_true = evaluate_model(
            model, name, X_2012, y_2012, scaler_dict[target], target, area_names_2012
        )

        predictions[name] = (y_pred, y_true)

except Exception as e:
    print(f"\n Ошибка при обучении на GPU: {e}")
    print(" Пробуем использовать CPU...")
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Повторная инициализация и обучение
    models = {
        "SimpleRNN": build_rnn_model(),
        "LSTM": build_lstm_model(),
        "GRU": build_gru_model()
    }

    predictions = {}

    for name, model in models.items():
        print(f"\nОбучение {name} на CPU...")
        model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1)

        y_pred, y_true = evaluate_model(
            model, name, X_2012, y_2012, scaler_dict[target], target, area_names_2012
        )

        predictions[name] = (y_pred, y_true)
        
        
import matplotlib.pyplot as plt

def plot_predictions(y_true, y_pred, areas, title):
    plt.figure(figsize=(14, 5))
    plt.plot(areas, y_true, label='Фактическое', marker='o')
    plt.plot(areas, y_pred, label='Предсказание', marker='x')
    plt.xticks(rotation=90)
    plt.title(title)
    plt.xlabel("Регион")
    plt.ylabel("Урожайность (hg/ha)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Построение графиков для каждой модели
for name, (y_pred, y_true) in predictions.items():
    plot_predictions(
        y_true.flatten(),
        y_pred.flatten(),
        area_names_2012,
        f"{name}: предсказание урожайности на 2012 год"
    )
