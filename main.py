import tensorflow as tf
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

rcParams["figure.figsize"] = 15, 7
sns.set(palette="Set2", font_scale=1.2)

pred_csv_path = "dataset/yield_df.csv"

x = pd.read_csv(pred_csv_path)
x.head()

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


# Model wrappers and grid search

rnn_param_grid = {
    'model__units': [32, 64],
    'model__dropout': [0.1, 0.2, 0.3],
    'model__lr': [0.001, 0.0005],
    'batch_size': [8, 16],
    'epochs': [50],
}
lstm_param_grid = {
    'model__units1': [32, 64],
    'model__units2': [16, 32],
    'model__dropout': [0.1, 0.2, 0.3],
    'model__lr': [0.001, 0.0005],
    'batch_size': [8, 16],
    'epochs': [50],
}
gru_param_grid = {
    'model__units1': [32, 64],
    'model__units2': [16, 32],
    'model__dropout': [0.1, 0.2, 0.3],
    'model__lr': [0.001, 0.0005],
    'batch_size': [8, 16],
    'epochs': [50],
}

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
    X, y = create_sequences_all(crop_data, features, target, look_back=look_back)
    X_rf = X.reshape(X.shape[0], -1)
    # Split for all models: 70% train, 30% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"Jumlah data training: {X_train.shape[0]}")
    print(f"Jumlah data testing: {X_test.shape[0]}")
    X_train_rf = X_train.reshape(X_train.shape[0], -1)
    X_test_rf = X_test.reshape(X_test.shape[0], -1)
    predictions = {}

    # Model Definitions
    print("[INFO] Mulai training SimpleRNN...")
    def build_rnn_model(units=64, dropout=0.2, lr=0.001):
        model = Sequential([
            tf.keras.Input(shape=(look_back, len(features))),
            SimpleRNN(units, activation='tanh'),
            Dropout(dropout),
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])
        return model
    print("[INFO] GridSearchCV SimpleRNN...")
    rnn_reg = KerasRegressor(model=build_rnn_model, verbose=0)
    rnn_grid = GridSearchCV(rnn_reg, rnn_param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=1)
    rnn_grid.fit(X_train, y_train)
    print("[INFO] SimpleRNN selesai training. Evaluasi...")
    rnn_best = rnn_grid.best_estimator_
    y_pred = rnn_best.predict(X_test)
    y_pred = scaler_dict[target].inverse_transform(y_pred.reshape(-1, 1))
    y_true = scaler_dict[target].inverse_transform(y_test.reshape(-1, 1))
    all_results.append({
        'item': item,
        'model': 'SimpleRNN',
        'category': 'Deep Learning',
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'mse': float(mean_squared_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
        'best_params': str(rnn_grid.best_params_)
    })
    print("[INFO] Evaluasi SimpleRNN selesai.")

    print("[INFO] Mulai training LSTM...")
    def build_lstm_model(units1=64, units2=32, dropout=0.2, lr=0.001):
        model = Sequential([
            tf.keras.Input(shape=(look_back, len(features))),
            LSTM(units1, activation='tanh', return_sequences=True),
            Dropout(dropout),
            LSTM(units2, activation='tanh'),
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])
        return model
    print("[INFO] GridSearchCV LSTM...")
    lstm_reg = KerasRegressor(model=build_lstm_model, verbose=0)
    lstm_grid = GridSearchCV(lstm_reg, lstm_param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=1)
    lstm_grid.fit(X_train, y_train)
    print("[INFO] LSTM selesai training. Evaluasi...")
    lstm_best = lstm_grid.best_estimator_
    y_pred = lstm_best.predict(X_test)
    y_pred = scaler_dict[target].inverse_transform(y_pred.reshape(-1, 1))
    y_true = scaler_dict[target].inverse_transform(y_test.reshape(-1, 1))
    all_results.append({
        'item': item,
        'model': 'LSTM',
        'category': 'Deep Learning',
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'mse': float(mean_squared_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
        'best_params': str(lstm_grid.best_params_)
    })
    print("[INFO] Evaluasi LSTM selesai.")

    print("[INFO] Mulai training GRU...")
    def build_gru_model(units1=64, units2=32, dropout=0.2, lr=0.001):
        model = Sequential([
            tf.keras.Input(shape=(look_back, len(features))),
            GRU(units1, activation='tanh', return_sequences=True),
            Dropout(dropout),
            GRU(units2),
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])
        return model
    print("[INFO] GridSearchCV GRU...")
    gru_reg = KerasRegressor(model=build_gru_model, verbose=0)
    gru_grid = GridSearchCV(gru_reg, gru_param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=1)
    gru_grid.fit(X_train, y_train)
    print("[INFO] GRU selesai training. Evaluasi...")
    gru_best = gru_grid.best_estimator_
    y_pred = gru_best.predict(X_test)
    y_pred = scaler_dict[target].inverse_transform(y_pred.reshape(-1, 1))
    y_true = scaler_dict[target].inverse_transform(y_test.reshape(-1, 1))
    all_results.append({
        'item': item,
        'model': 'GRU',
        'category': 'Deep Learning',
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'mse': float(mean_squared_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
        'best_params': str(gru_grid.best_params_)
    })
    print("[INFO] Evaluasi GRU selesai.")

    print("[INFO] Mulai training RandomForest...")
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
    rf = RandomForestRegressor(random_state=42)
    print("[INFO] GridSearchCV RandomForest...")
    rf_grid = GridSearchCV(rf, rf_param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    rf_grid.fit(X_train_rf, y_train)
    print("[INFO] RandomForest selesai training. Evaluasi...")
    rf_best = rf_grid.best_estimator_
    y_pred = rf_best.predict(X_test_rf)
    y_pred = scaler_dict[target].inverse_transform(y_pred.reshape(-1, 1))
    y_true = scaler_dict[target].inverse_transform(y_test.reshape(-1, 1))
    all_results.append({
        'item': item,
        'model': 'RandomForest',
        'category': 'Tree Ensemble',
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'mse': float(mean_squared_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
        'best_params': str(rf_grid.best_params_)
    })
    print("[INFO] Evaluasi RandomForest selesai.")

    print("[INFO] Mulai training BaggingRegressor...")
    bag_param_grid = {
        'estimator': [RandomForestRegressor(max_depth=3), RandomForestRegressor(max_depth=5)],
        'n_estimators': [10, 20, 50]
    }
    bag = BaggingRegressor(random_state=42)
    print("[INFO] GridSearchCV BaggingRegressor...")
    bag_grid = GridSearchCV(bag, bag_param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    bag_grid.fit(X_train_rf, y_train)
    print("[INFO] BaggingRegressor selesai training. Evaluasi...")
    bag_best = bag_grid.best_estimator_
    y_pred = bag_best.predict(X_test_rf)
    y_pred = scaler_dict[target].inverse_transform(y_pred.reshape(-1, 1))
    y_true = scaler_dict[target].inverse_transform(y_test.reshape(-1, 1))
    all_results.append({
        'item': item,
        'model': 'BaggingRegressor',
        'category': 'Tree Ensemble',
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'mse': float(mean_squared_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
        'best_params': str(bag_grid.best_params_)
    })
    print("[INFO] Evaluasi BaggingRegressor selesai.")

    print("[INFO] Mulai training GradientBoostingRegressor...")
    gb_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10],
        'learning_rate': [0.01, 0.05, 0.1]
    }
    gb = GradientBoostingRegressor(random_state=42)
    print("[INFO] GridSearchCV GradientBoostingRegressor...")
    gb_grid = GridSearchCV(gb, gb_param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    gb_grid.fit(X_train_rf, y_train)
    print("[INFO] GradientBoostingRegressor selesai training. Evaluasi...")
    gb_best = gb_grid.best_estimator_
    y_pred = gb_best.predict(X_test_rf)
    y_pred = scaler_dict[target].inverse_transform(y_pred.reshape(-1, 1))
    y_true = scaler_dict[target].inverse_transform(y_test.reshape(-1, 1))
    all_results.append({
        'item': item,
        'model': 'GradientBoosting',
        'category': 'Boosting',
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'mse': float(mean_squared_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
        'best_params': str(gb_grid.best_params_)
    })
    print("[INFO] Evaluasi GradientBoostingRegressor selesai.")


# Evaluation Visualization
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

    # Save model evaluation results to CSV
csv_path = os.path.join(results_dir, 'model_results_per_item_gridsearch.csv')
all_results_sorted = sorted(all_results, key=lambda x: (x['item'], x['mae']))
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Item', 'Model', 'Category', 'MAE', 'MSE', 'R2', 'Best_Params'])
    for res in all_results_sorted:
        writer.writerow([res['item'], res['model'], res['category'], res['mae'], res['mse'], res['r2'], res['best_params']])


# (All year-specific and area_names_2012 code removed. All results and visualizations are now based on historical test split only.)
