from warnings import filterwarnings

filterwarnings("ignore")

from datetime import datetime
import typing as tp
from IPython.display import display
from pylab import rcParams

import matplotlib.pyplot as plt
import json
import seaborn as sns

rcParams["figure.figsize"] = 15, 7 
sns.set(palette="Set2", font_scale=1.2)  

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_squared_error,      
    mean_absolute_error,     
    mean_absolute_percentage_error, 
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


data = pd.read_csv(pred_csv_path)
maize_data = data[data['Item'] == 'Maize'].dropna().sort_values(['Area', 'Year'])
print(f"Исходный размер датасета: {data.shape}")

def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    mask = (df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)
    return df[mask]

maize_data = remove_outliers(maize_data, 'hg/ha_yield')


maize_data['log_yield'] = np.log(maize_data['hg/ha_yield'] + 1)

maize_data['rolling_yield_mean3'] = maize_data.groupby('Area')['hg/ha_yield'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

scaler_rolling = MinMaxScaler()
maize_data['rolling_yield_mean3'] = scaler_rolling.fit_transform(maize_data[['rolling_yield_mean3']])

features = ['avg_temp', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'rolling_yield_mean3']
target = 'log_yield'

scaler_dict = {}
for col in features:
    scaler = MinMaxScaler()
    maize_data[col] = scaler.fit_transform(maize_data[[col]])
    scaler_dict[col] = scaler
target_scaler = MinMaxScaler()
maize_data[target] = target_scaler.fit_transform(maize_data[[target]])
scaler_dict[target] = target_scaler

look_back = 9

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
                    continue  

                X.append(seq[features].values)
                y.append(target_val)
                areas.append(area)

    return np.array(X), np.array(y), areas

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

def build_rnn_model():
    model = Sequential([
        SimpleRNN(64, activation='tanh', input_shape=(look_back, len(features))),
        Dropout(0.2),
        Dense(64, activation='relu', kernel_regularizer='l2'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


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


from tensorflow.keras.callbacks import ReduceLROnPlateau
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-5)

models = {
    "SimpleRNN": build_rnn_model(),
    "LSTM": build_lstm_model(),
    "GRU": build_gru_model()
}
predictions = {}

for name, model in models.items():
    print(f"\nОбучение {name}...")
    model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1, validation_split=0.1, callbacks=[early_stop, reduce_lr])
    y_pred, y_true = evaluate_model(
        model, name, X_2012, y_2012, scaler_dict[target], target, area_names_2012
    )
    predictions[name] = (y_pred, y_true)

from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor

X_rf = X.reshape(X.shape[0], -1)
X_2012_rf = X_2012.reshape(X_2012.shape[0], -1)

tscv = TimeSeriesSplit(n_splits=5)
rf = RandomForestRegressor(random_state=42)
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}
rf_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=5, cv=tscv, n_jobs=-1, random_state=42)
rf_search.fit(X_rf, y)
rf_best = rf_search.best_estimator_
rf_pred_scaled = rf_best.predict(X_2012_rf)
rf_pred = scaler_dict[target].inverse_transform(rf_pred_scaled.reshape(-1, 1))
rf_true = scaler_dict[target].inverse_transform(y_2012.reshape(-1, 1))
from sklearn.metrics import r2_score
print(f"\n== Метрики для RandomForest на 2012 год ==")
print(f"MAE: {mean_absolute_error(rf_true, rf_pred):.2f} hg/ha")
print(f"MSE: {mean_squared_error(rf_true, rf_pred):.2f} hg/ha")
print(f"R² : {r2_score(rf_true, rf_pred):.4f}")
predictions['RandomForest'] = (rf_pred, rf_true)

all_preds = np.column_stack([pred[0].flatten() for pred in predictions.values()])
ensemble_pred = np.mean(all_preds, axis=1)
ensemble_true = list(predictions.values())[0][1].flatten() 
print(f"\n== Метрики для Ensemble (Average) на 2012 год ==")
print(f"MAE: {mean_absolute_error(ensemble_true, ensemble_pred):.2f} hg/ha")
print(f"MSE: {mean_squared_error(ensemble_true, ensemble_pred):.2f} hg/ha")
print(f"R² : {r2_score(ensemble_true, ensemble_pred):.4f}")
predictions['Ensemble'] = (ensemble_pred, ensemble_true)
        
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

results = {}
for name, (y_pred, y_true) in predictions.items():
    y_pred = y_pred.flatten().tolist()
    y_true = y_true.flatten().tolist()
    error = [yt - yp for yt, yp in zip(y_true, y_pred)]
    results[name] = {
        'y_true': y_true,
        'y_pred': y_pred,
        'error': error,
        'area_names': list(area_names_2012),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'mse': float(mean_squared_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
    }

plt.figure(figsize=(16, 7))
for name in predictions:
    plt.plot(area_names_2012, results[name]['error'], marker='o', label=f'Error {name}')
plt.xticks(rotation=90)
plt.title('Error per Area untuk Semua Model')
plt.xlabel('Area')
plt.ylabel('Error (y_true - y_pred)')
plt.legend()
plt.tight_layout()
plt.show()

if 'RandomForest' in predictions:
    try:
        importances = rf_best.feature_importances_.tolist()
        results['RandomForest']['feature_importance'] = importances
    except Exception as e:
        results['RandomForest']['feature_importance'] = str(e)


with open('model_results.json', 'w') as f:
    json.dump(results, f, indent=2)

import csv
import os
output_dir = 'evaluated_model'
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, 'model_comparison.csv')
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Model', 'MAE', 'MSE', 'R2'])
    for name, res in results.items():
        writer.writerow([name, res['mae'], res['mse'], res['r2']])

for name, (y_pred, y_true) in predictions.items():
    plot_predictions(
        y_true.flatten(),
        y_pred.flatten(),
        area_names_2012,
        f"{name}: предсказание урожайности на 2012 год"
    )
