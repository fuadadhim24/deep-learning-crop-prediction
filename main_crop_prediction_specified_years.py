from warnings import filterwarnings

filterwarnings("ignore")

from datetime import datetime
import typing as tp
from IPython.display import display
from pylab import rcParams


import matplotlib.pyplot as plt
import json
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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



from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor

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
print(f"Исходный размер датасета: {data.shape}")

def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    mask = (df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)
    return df[mask]

features = ['avg_temp', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'rolling_yield_mean3']
target = 'log_yield'
look_back = 9

all_results = []


from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

for item in data['Item'].unique():
    print(f"\n=== Mulai proses untuk crop: {item} ===")
    crop_data = data[data['Item'] == item].dropna().sort_values(['Area', 'Year'])
    crop_data = remove_outliers(crop_data, 'hg/ha_yield')
    crop_data['log_yield'] = np.log(crop_data['hg/ha_yield'] + 1)
    crop_data['rolling_yield_mean3'] = crop_data.groupby('Area')['hg/ha_yield'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    scaler_rolling = MinMaxScaler()
    crop_data['rolling_yield_mean3'] = scaler_rolling.fit_transform(crop_data[['rolling_yield_mean3']])
    scaler_dict = {}
    for col in features:
        scaler = MinMaxScaler()
        crop_data[col] = scaler.fit_transform(crop_data[[col]])
        scaler_dict[col] = scaler
    target_scaler = MinMaxScaler()
    crop_data[target] = target_scaler.fit_transform(crop_data[[target]])
    scaler_dict[target] = target_scaler

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

    def create_sequences_all(data, features, target, look_back=9):
        X, y = [], []
        areas = data['Area'].unique()
        for area in areas:
            region_data = data[data['Area'] == area].sort_values('Year')
            for i in range(look_back, len(region_data)):
                X.append(region_data[features].iloc[i-look_back:i].values)
                y.append(region_data[target].iloc[i])
        return np.array(X), np.array(y)

    print("Membuat sequence data untuk training dan testing...")
    X_2012, y_2012, area_names_2012 = create_test_sequences_for_year(crop_data, features, target, 2012, look_back=look_back)
    X, y = create_sequences_all(crop_data, features, target, look_back=look_back)
    X_rf = X.reshape(X.shape[0], -1)
    X_2012_rf = X_2012.reshape(X_2012.shape[0], -1)
    # Split for all models: 70% train, 30% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"Jumlah data training: {X_train.shape[0]}")
    print(f"Jumlah data testing: {X_test.shape[0]}")
    X_train_rf = X_train.reshape(X_train.shape[0], -1)
    X_test_rf = X_test.reshape(X_test.shape[0], -1)

    predictions = {}

    print("GridSearchCV: SimpleRNN...")
    def build_rnn_model(units=64, dropout=0.2, lr=0.001):
        model = Sequential([
            SimpleRNN(units, activation='tanh', input_shape=(look_back, len(features))),
            Dropout(dropout),
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])
        return model

    rnn_param_grid = {
        'model__units': [32, 64],
        'model__dropout': [0.1, 0.2, 0.3],
        'model__lr': [0.001, 0.0005],
        'batch_size': [8, 16],
        'epochs': [50],
    }
    rnn_reg = KerasRegressor(model=build_rnn_model, verbose=0)
    rnn_grid = GridSearchCV(rnn_reg, rnn_param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=1)
    rnn_grid.fit(X_train, y_train)
    rnn_best = rnn_grid.best_estimator_
    y_pred = rnn_best.predict(X_test)
    y_pred = scaler_dict[target].inverse_transform(y_pred.reshape(-1, 1))
    y_true = scaler_dict[target].inverse_transform(y_test.reshape(-1, 1))
    predictions['SimpleRNN_best'] = (y_pred, y_true, rnn_grid.best_params_)

    print("GridSearchCV: LSTM...")
    def build_lstm_model(units1=64, units2=32, dropout=0.2, lr=0.001):
        model = Sequential([
            LSTM(units1, activation='tanh', return_sequences=True, input_shape=(look_back, len(features))),
            Dropout(dropout),
            LSTM(units2, activation='tanh'),
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])
        return model

    lstm_param_grid = {
        'model__units1': [32, 64],
        'model__units2': [16, 32],
        'model__dropout': [0.1, 0.2, 0.3],
        'model__lr': [0.001, 0.0005],
        'batch_size': [8, 16],
        'epochs': [50],
    }
    lstm_reg = KerasRegressor(model=build_lstm_model, verbose=0)
    lstm_grid = GridSearchCV(lstm_reg, lstm_param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=1)
    lstm_grid.fit(X_train, y_train)
    lstm_best = lstm_grid.best_estimator_
    y_pred = lstm_best.predict(X_test)
    y_pred = scaler_dict[target].inverse_transform(y_pred.reshape(-1, 1))
    y_true = scaler_dict[target].inverse_transform(y_test.reshape(-1, 1))
    predictions['LSTM_best'] = (y_pred, y_true, lstm_grid.best_params_)

    print("GridSearchCV: GRU...")
    def build_gru_model(units1=64, units2=32, dropout=0.2, lr=0.001):
        model = Sequential([
            GRU(units1, activation='tanh', return_sequences=True, input_shape=(look_back, len(features))),
            Dropout(dropout),
            GRU(units2),
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])
        return model

    gru_param_grid = {
        'model__units1': [32, 64],
        'model__units2': [16, 32],
        'model__dropout': [0.1, 0.2, 0.3],
        'model__lr': [0.001, 0.0005],
        'batch_size': [8, 16],
        'epochs': [50],
    }
    gru_reg = KerasRegressor(model=build_gru_model, verbose=0)
    gru_grid = GridSearchCV(gru_reg, gru_param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=1)
    gru_grid.fit(X_train, y_train)
    gru_best = gru_grid.best_estimator_
    y_pred = gru_best.predict(X_test)
    y_pred = scaler_dict[target].inverse_transform(y_pred.reshape(-1, 1))
    y_true = scaler_dict[target].inverse_transform(y_test.reshape(-1, 1))
    predictions['GRU_best'] = (y_pred, y_true, gru_grid.best_params_)

    print("GridSearchCV: RandomForest...")
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
    rf = RandomForestRegressor(random_state=42)
    rf_grid = GridSearchCV(rf, rf_param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    rf_grid.fit(X_train_rf, y_train)
    rf_best = rf_grid.best_estimator_
    rf_pred_scaled = rf_best.predict(X_test_rf)
    rf_pred = scaler_dict[target].inverse_transform(rf_pred_scaled.reshape(-1, 1))
    rf_true = scaler_dict[target].inverse_transform(y_test.reshape(-1, 1))
    predictions['RandomForest_best'] = (rf_pred, rf_true, rf_grid.best_params_)

    print("GridSearchCV: BaggingRegressor...")
    bag_param_grid = {
        'estimator': [DecisionTreeRegressor(max_depth=3), DecisionTreeRegressor(max_depth=5)],
        'n_estimators': [10, 20, 50]
    }
    bag = BaggingRegressor(random_state=42)
    bag_grid = GridSearchCV(bag, bag_param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    bag_grid.fit(X_train_rf, y_train)
    bag_best = bag_grid.best_estimator_
    bag_pred_scaled = bag_best.predict(X_test_rf)
    bag_pred = scaler_dict[target].inverse_transform(bag_pred_scaled.reshape(-1, 1))
    bag_true = scaler_dict[target].inverse_transform(y_test.reshape(-1, 1))
    predictions['BaggingRegressor_best'] = (bag_pred, bag_true, bag_grid.best_params_)

    print("GridSearchCV: GradientBoostingRegressor...")
    gb_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10],
        'learning_rate': [0.01, 0.05, 0.1]
    }
    gb = GradientBoostingRegressor(random_state=42)
    gb_grid = GridSearchCV(gb, gb_param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    gb_grid.fit(X_train_rf, y_train)
    gb_best = gb_grid.best_estimator_
    gb_pred_scaled = gb_best.predict(X_test_rf)
    gb_pred = scaler_dict[target].inverse_transform(gb_pred_scaled.reshape(-1, 1))
    gb_true = scaler_dict[target].inverse_transform(y_test.reshape(-1, 1))
    predictions['GradientBoosting_best'] = (gb_pred, gb_true, gb_grid.best_params_)

    print(f"Selesai semua GridSearchCV untuk crop: {item}")

    # After predictions dict is filled:
    for name, (y_pred, y_true, best_params) in predictions.items():
        y_pred = y_pred.flatten().tolist()
        y_true = y_true.flatten().tolist()
        error = [yt - yp for yt, yp in zip(y_true, y_pred)]
        # Determine category
        if 'SimpleRNN' in name:
            category = 'Deep Learning'
        elif 'LSTM' in name:
            category = 'Deep Learning'
        elif 'GRU' in name:
            category = 'Deep Learning'
        elif 'RandomForest' in name:
            category = 'Tree Ensemble'
        elif 'Bagging' in name:
            category = 'Tree Ensemble'
        elif 'GradientBoosting' in name:
            category = 'Boosting'
        else:
            category = 'Other'
        all_results.append({
            'item': item,
            'model': name,
            'category': category,
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'mse': float(mean_squared_error(y_true, y_pred)),
            'r2': float(r2_score(y_true, y_pred)),
            'best_params': str(best_params)
        })


# === VISUALIZATION & EXPORT ===
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns

results_dir = 'results/all_models'
os.makedirs(results_dir, exist_ok=True)

# 1. Histogram Yield
plt.figure(figsize=(10,6))
sns.histplot(data['hg/ha_yield'], bins=30, kde=True, color='skyblue')
plt.title('Distribusi Yield (hg/ha_yield)')
plt.xlabel('Yield (hg/ha)')
plt.ylabel('Frekuensi')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, '1_hist_yield_distribution.png'))
plt.close()

# 2. Korelasi Heatmap
plt.figure(figsize=(10,8))
num_cols = data.select_dtypes(include='number').columns
corr = data[num_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title('Heatmap Korelasi Fitur Numerik')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, '2_heatmap_korelasi.png'))
plt.close()

# 3. Jumlah Data per Area (Encoded)
plt.figure(figsize=(14,6))
area_counts = data['Area'].value_counts().sort_values(ascending=False)
sns.barplot(x=area_counts.index, y=area_counts.values, palette='viridis')
plt.title('Jumlah Data per Area')
plt.xlabel('Area (Encoded)')
plt.ylabel('Jumlah Data')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, '3_countplot_area.png'))
plt.close()

# 4. Boxplot Yield per Crop
plt.figure(figsize=(12,6))
sns.boxplot(x='Item', y='hg/ha_yield', data=data, palette='Set2')
plt.title('Boxplot Yield per Crop')
plt.xlabel('Crop (Item)')
plt.ylabel('Yield (hg/ha)')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, '4_boxplot_yield_per_crop.png'))
plt.close()

# 5. R² Score Tiap Model per Crop
import pandas as pd
df_results = pd.DataFrame(all_results)
plt.figure(figsize=(14,7))
sns.barplot(data=df_results, x='item', y='r2', hue='model', ci=None)
plt.title('R² Score Tiap Model per Crop')
plt.ylabel('R² Score')
plt.xlabel('Crop (Item)')
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, '5_r2score_per_model_crop.png'))
plt.close()

# 6. Rata-rata R² Score per Model
plt.figure(figsize=(10,6))
avg_r2 = df_results.groupby('model')['r2'].mean().sort_values(ascending=False)
sns.barplot(x=avg_r2.index, y=avg_r2.values, palette='Blues_d')
plt.title('Rata-rata R² Score per Model')
plt.ylabel('Rata-rata R²')
plt.xlabel('Model')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, '6_avg_r2_score_per_model.png'))
plt.close()

# 7. Rata-rata MAE per Model
plt.figure(figsize=(10,6))
avg_mae = df_results.groupby('model')['mae'].mean().sort_values()
sns.barplot(x=avg_mae.index, y=avg_mae.values, palette='Reds')
plt.title('Rata-rata MAE per Model')
plt.ylabel('Rata-rata MAE')
plt.xlabel('Model')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, '7_avg_mae_per_model.png'))
plt.close()

# Simpan hasil evaluasi model ke CSV
csv_path = os.path.join(results_dir, 'model_results_per_item_gridsearch.csv')
all_results_sorted = sorted(all_results, key=lambda x: (x['item'], x['mae']))
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Item', 'Model', 'Category', 'MAE', 'MSE', 'R2', 'Best_Params'])
    for res in all_results_sorted:
        writer.writerow([res['item'], res['model'], res['category'], res['mae'], res['mse'], res['r2'], res['best_params']])

def build_rnn_model():
    model = Sequential([
        SimpleRNN(64, activation='tanh', input_shape=(look_back, len(features))),
        Dropout(0.2),
        Dense(64, activation='relu', kernel_regularizer='l2'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model



# Tuning function for LSTM
def build_lstm_model(units1=64, units2=32, dropout=0.2, lr=0.001):
    model = Sequential([
        LSTM(units1, activation='tanh', return_sequences=True, input_shape=(look_back, len(features))),
        Dropout(dropout),
        LSTM(units2, activation='tanh'),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])
    return model

# Tuning function for GRU
def build_gru_model(units1=64, units2=32, dropout=0.2, lr=0.001):
    model = Sequential([
        GRU(units1, activation='tanh', return_sequences=True, input_shape=(look_back, len(features))),
        Dropout(dropout),
        GRU(units2),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])
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

    print(f"\n== Metrics for {name} in 2012 ==")
    print(f"MAE: {mae:.2f} hg/ha")
    print(f"MSE: {mse:.2f} hg/ha")
    print(f"R² : {r2:.4f}")
    return y_pred, y_true


from tensorflow.keras.callbacks import ReduceLROnPlateau
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-5)

# Automated GridSearchCV for deep learning models using KerasRegressor
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV

predictions = {}

# SimpleRNN
def build_rnn_model_sklearn(units=64, dropout=0.2, lr=0.001):
    model = Sequential([
        SimpleRNN(units, activation='tanh', input_shape=(look_back, len(features))),
        Dropout(dropout),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])
    return model

rnn_param_grid = {
    'units': [32, 64],
    'dropout': [0.1, 0.2, 0.3],
    'lr': [0.001, 0.0005],
    'batch_size': [8, 16],
    'epochs': [50],
}
rnn_reg = KerasRegressor(model=build_rnn_model_sklearn, verbose=0)
rnn_grid = GridSearchCV(rnn_reg, rnn_param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=1)
rnn_grid.fit(X_train, y_train)
rnn_best = rnn_grid.best_estimator_
y_pred = rnn_best.predict(X_2012)
y_pred = scaler_dict[target].inverse_transform(y_pred.reshape(-1, 1))
y_true = scaler_dict[target].inverse_transform(y_2012.reshape(-1, 1))
predictions['SimpleRNN_best'] = (y_pred, y_true)

# LSTM
def build_lstm_model_sklearn(units1=64, units2=32, dropout=0.2, lr=0.001):
    model = Sequential([
        LSTM(units1, activation='tanh', return_sequences=True, input_shape=(look_back, len(features))),
        Dropout(dropout),
        LSTM(units2, activation='tanh'),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])
    return model

lstm_param_grid = {
    'units1': [32, 64],
    'units2': [16, 32],
    'dropout': [0.1, 0.2, 0.3],
    'lr': [0.001, 0.0005],
    'batch_size': [8, 16],
    'epochs': [50],
}
lstm_reg = KerasRegressor(model=build_lstm_model_sklearn, verbose=0)
lstm_grid = GridSearchCV(lstm_reg, lstm_param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=1)
lstm_grid.fit(X_train, y_train)
lstm_best = lstm_grid.best_estimator_
y_pred = lstm_best.predict(X_2012)
y_pred = scaler_dict[target].inverse_transform(y_pred.reshape(-1, 1))
y_true = scaler_dict[target].inverse_transform(y_2012.reshape(-1, 1))
predictions['LSTM_best'] = (y_pred, y_true)

# GRU
def build_gru_model_sklearn(units1=64, units2=32, dropout=0.2, lr=0.001):
    model = Sequential([
        GRU(units1, activation='tanh', return_sequences=True, input_shape=(look_back, len(features))),
        Dropout(dropout),
        GRU(units2),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])
    return model

gru_param_grid = {
    'units1': [32, 64],
    'units2': [16, 32],
    'dropout': [0.1, 0.2, 0.3],
    'lr': [0.001, 0.0005],
    'batch_size': [8, 16],
    'epochs': [50],
}
gru_reg = KerasRegressor(model=build_gru_model_sklearn, verbose=0)
gru_grid = GridSearchCV(gru_reg, gru_param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=1)
gru_grid.fit(X_train, y_train)
gru_best = gru_grid.best_estimator_
y_pred = gru_best.predict(X_2012)
y_pred = scaler_dict[target].inverse_transform(y_pred.reshape(-1, 1))
y_true = scaler_dict[target].inverse_transform(y_2012.reshape(-1, 1))
predictions['GRU_best'] = (y_pred, y_true)


# RandomForest with GridSearchCV
from sklearn.model_selection import GridSearchCV
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}
rf = RandomForestRegressor(random_state=42)
rf_grid = GridSearchCV(rf, rf_param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
rf_grid.fit(X_rf, y)
rf_best = rf_grid.best_estimator_
rf_pred_scaled = rf_best.predict(X_2012_rf)
rf_pred = scaler_dict[target].inverse_transform(rf_pred_scaled.reshape(-1, 1))
rf_true = scaler_dict[target].inverse_transform(y_2012.reshape(-1, 1))
predictions['RandomForest_best'] = (rf_pred, rf_true)


# BaggingRegressor with GridSearchCV
from sklearn.tree import DecisionTreeRegressor
bag_param_grid = {
    'base_estimator': [DecisionTreeRegressor(max_depth=3), DecisionTreeRegressor(max_depth=5)],
    'n_estimators': [10, 20, 50]
}
bag = BaggingRegressor(random_state=42)
bag_grid = GridSearchCV(bag, bag_param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
bag_grid.fit(X_rf, y)
bag_best = bag_grid.best_estimator_
bag_pred_scaled = bag_best.predict(X_2012_rf)
bag_pred = scaler_dict[target].inverse_transform(bag_pred_scaled.reshape(-1, 1))
bag_true = scaler_dict[target].inverse_transform(y_2012.reshape(-1, 1))
predictions['BaggingRegressor_best'] = (bag_pred, bag_true)

# GradientBoostingRegressor with GridSearchCV
gb_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.05, 0.1]
}
gb = GradientBoostingRegressor(random_state=42)
gb_grid = GridSearchCV(gb, gb_param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
gb_grid.fit(X_rf, y)
gb_best = gb_grid.best_estimator_
gb_pred_scaled = gb_best.predict(X_2012_rf)
gb_pred = scaler_dict[target].inverse_transform(gb_pred_scaled.reshape(-1, 1))
gb_true = scaler_dict[target].inverse_transform(y_2012.reshape(-1, 1))
predictions['GradientBoosting_best'] = (gb_pred, gb_true)

# Note: For deep learning models (SimpleRNN, LSTM, GRU),
# you can use KerasRegressor with GridSearchCV for full automation.
# This is not implemented here for brevity and stability.




        
import matplotlib.pyplot as plt


def plot_predictions(y_true, y_pred, areas, title):
    plt.figure(figsize=(14, 5))
    plt.plot(areas, y_true, label='Actual', marker='o')
    plt.plot(areas, y_pred, label='Prediction', marker='x')
    plt.xticks(rotation=90)
    plt.title(title)
    plt.xlabel("Region")
    plt.ylabel("Yield (hg/ha)")
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
        f"{name}: yield prediction for 2012"
    )
