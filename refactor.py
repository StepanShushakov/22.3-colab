import pandas as pd
cars = pd.read_csv('japan_cars_dataset.csv', sep=',')

# Удалим строки с пустыми значениями
cars = cars.dropna()

from sklearn.model_selection import train_test_split
from datetime import datetime
import numpy as np
from keras import utils
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Input, concatenate, Embedding, GlobalAveragePooling1D
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from keras.metrics import Metric
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib
matplotlib.use('TkAgg')  # или 'Qt5Agg', 'MacOSX' и т.д.
import matplotlib.pyplot as plt
# Токенизатор
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Удаляем первую колонку без имени
cars = cars.drop(columns=cars.columns[0])

# Определяем японские марки
japan_brands = ['toyota', 'nissan', 'honda', 'mazda', 'suzuki', 'mitsubishi', 'daihatsu', 'subaru', 'isuzu', 'hino', 'mitsuoka', 'kubota']

# Удаляем не японские марки
rows_before = len(cars)
cars = cars[cars['mark'].isin(japan_brands)]
removed_non_japan = rows_before - len(cars)
print(f"Удалено не японских марок: {removed_non_japan}")

# Удаляем редкие марки (менее 4 вхождений)
rows_before = len(cars)
rare_marks = cars['mark'].value_counts()[lambda s: s < 4].index
cars = cars[~cars['mark'].isin(rare_marks)]
removed_rare = rows_before - len(cars)
print(f"Удалено редких марок (менее 4 вхождений): {removed_rare}")

print(f"Осталось строк в датасете: {len(cars)}")


# Удаление выбросов (outliers) для улучшения предсказания цены
print("\n=== Удаление выбросов ===")
rows_before_outliers = len(cars)

# Функция для обнаружения выбросов методом IQR (межквартильный размах)
def remove_outliers_iqr(df, columns, multiplier=1.5):
    """
    Удаляет выбросы из датафрейма по указанным колонкам
    multiplier: коэффициент для IQR (1.5 - стандарт, больше - мягче фильтрация)
    """
    df_clean = df.copy()

    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        outliers_count = len(df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)])
        print(f"{col}: диапазон [{lower_bound:.2f}, {upper_bound:.2f}], выбросов: {outliers_count}")

        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

    return df_clean

# Определяем колонки для проверки на выбросы
numeric_columns = ['price'] #оставляем только цену, что бы не терять вариативность 'year', 'mileage', 'engine_capacity']

# Удаляем выбросы
cars = remove_outliers_iqr(cars, numeric_columns, multiplier=1.5)

removed_outliers = rows_before_outliers - len(cars)
print(f"\nВсего удалено строк с выбросами: {removed_outliers}")
print(f"Осталось строк в датасете: {len(cars)}")
print(f"Процент удаленных данных: {(removed_outliers/rows_before_outliers)*100:.2f}%")

# Объединяем mark и model в один столбец mark_model и удаляем исходные
# cars['mark_model'] = cars['mark'].astype(str) + ' ' + cars['model'].astype(str)
# cars = cars.drop(columns=['mark', 'model'])
#Перемешаем датафрейм
cars = cars.sample(frac=1, random_state=42).reset_index(drop=True)

# Используется встроенный в Keras токенизатор для разбиения текста и построения частотного словаря
tokenizer_mark = Tokenizer(
    num_words=18,                                            # оптимальный размер словаря для марки
    filters='!"«»#$№%&()*+,-–—./:;<=>?@[\\]^_`{|}~\t\n\xa0', # убираемые из текста ненужные символы
    lower=True,                                              # приведение слов к нижнему регистру
    split=' ',                                               # разделитель слов
    oov_token='unknown',                                     # токен для слов, которые не вошли в словарь
    char_level=False                                         # разделяем по словам, а не по единичным символам
)

tokenizer_model = Tokenizer(
    num_words=149,                                           # оптимальный размер словаря для модели
    filters='!"«»#$№%&()*+,-–—./:;<=>?@[\\]^_`{|}~\t\n\xa0', # убираемые из текста ненужные символы
    lower=True,                                              # приведение слов к нижнему регистру
    split=' ',                                               # разделитель слов
    oov_token='unknown',                                     # токен для слов, которые не вошли в словарь
    char_level=False                                         # разделяем по словам, а не по единичным символам
)

transmission_class  = {'mt': 0,
                       'at': 1
                       }
drive_class         = {'2wd': 0,
                       '4wd': 1
                       }
hand_drive_class    = {'lhd': 0,
                       'rhd': 1
                       }
fuel_class          = {'hybrid': 0,
                       'lpg': 1,
                       'diesel': 2,
                       'gasoline': 3
                       }

# Удалена функция range2OHE, так как числовые признаки теперь остаются числовыми

# Общая функция преобразования строки к multi-вектору
# На входе данные и словарь сопоставления подстрок классам
def str2multiOHE(param, class_dict):
    # Определение размерности выходного вектора, как число уникальных классов
    num_classes = len(set(class_dict.values()))
    # Создание нулевого вектора
    result = np.zeros(num_classes)
    # Если не смогли распарсить, то поле не заполнено
    # Устанавливаем значение по умолчанию (последний элемент в словаре)
    if not isinstance(param, str):
        param = list(class_dict.keys())[-1]
    # Поиск значения в словаре и, если нашли, то проставляем 1 в найденной позиции
    for value, cls in class_dict.items():
        if value in param:
            result[cls] = 1.
    return result

# Обучающая выборка по числовым данным
# Фиксация индексов столбцов
COL_YEAR            = cars.columns.get_loc('year')
COL_MILEAGE         = cars.columns.get_loc('mileage')
COL_ENGINE_CAPACITY = cars.columns.get_loc('engine_capacity')
COL_TRANSMISSION    = cars.columns.get_loc('transmission')
COL_DRIVE           = cars.columns.get_loc('drive')
COL_HAND_DRIVE      = cars.columns.get_loc('hand_drive')
COL_FUEL            = cars.columns.get_loc('fuel')
COL_PRICE           = cars.columns.get_loc('price')

def get_row_data(row):
    # Базовые признаки
    year = float(row[COL_YEAR])
    mileage = float(row[COL_MILEAGE])
    engine_capacity = float(row[COL_ENGINE_CAPACITY])

    # Улучшенные признаки: производные признаки с большим смыслом
    current_year = datetime.now().year
    car_age = current_year - year  # возраст автомобиля
    mileage_per_year = mileage / (car_age + 1)  # пробег на год (избегаем деления на 0)
    log_mileage = np.log1p(mileage)  # логарифм пробега для нормализации распределения

    # Объединение всех входных данных в один общий вектор
    x_data = np.hstack([
        np.array([float(row[COL_YEAR])]),
        np.array([float(row[COL_MILEAGE])]),
        np.array([float(row[COL_ENGINE_CAPACITY])]),
        str2multiOHE(row[COL_TRANSMISSION], transmission_class),
        str2multiOHE(row[COL_DRIVE], drive_class),
        str2multiOHE(row[COL_HAND_DRIVE], hand_drive_class),
        str2multiOHE(row[COL_FUEL], fuel_class)
    ])

    # Вектор цен в исходных единицах
    y_data = np.array([row[COL_PRICE]])


    return x_data, y_data

def get_train_data(dataFrame):
    x_data = []
    y_data = []

    for row in dataFrame.values:
        x, y = get_row_data(row)
        x_data.append(x)
        y_data.append(y)

    return np.array(x_data), np.array(y_data)

# Формирование обучающей выборки из загруженного набора данных
x_data, y_data = get_train_data(cars)

# === Ключевое исправление: сохраняем индексы разбиения ===
indices = np.arange(len(cars))
train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42, shuffle=True)
train_idx, val_idx = train_test_split(train_idx, test_size=0.176, random_state=42, shuffle=True)

# Теперь создаём разбиение с учётом индексов
x_train, x_test = x_data[train_idx], x_data[test_idx]
y_train, y_test = y_data[train_idx], y_data[test_idx]
x_train, x_val = x_data[train_idx], x_data[val_idx]
y_train, y_val = y_data[train_idx], y_data[val_idx]

# Теперь можно правильно взять текстовые данные
mark_train = cars['mark'].iloc[train_idx].values
model_train = cars['model'].iloc[train_idx].values
mark_val = cars['mark'].iloc[val_idx].values
model_val = cars['model'].iloc[val_idx].values
mark_test = cars['mark'].iloc[test_idx].values
model_test = cars['model'].iloc[test_idx].values

# Теперь обучаем токенизаторы ТОЛЬКО на train
tokenizer_mark.fit_on_texts(mark_train)
tokenizer_model.fit_on_texts(model_train)

# Преобразуем тексты
mark_seq_train = tokenizer_mark.texts_to_sequences(mark_train)
model_seq_train = tokenizer_model.texts_to_sequences(model_train)
mark_seq_val = tokenizer_mark.texts_to_sequences(mark_val)
model_seq_val = tokenizer_model.texts_to_sequences(model_val)
mark_seq_test = tokenizer_mark.texts_to_sequences(mark_test)
model_seq_test = tokenizer_model.texts_to_sequences(model_test)

# Далее — как было: pad_sequences и т.д.
max_len_mark = min(max(len(seq) for seq in mark_seq_train or [[1]]), 10)
max_len_model = min(max(len(seq) for seq in model_seq_train or [[1]]), 15)

x_train_mark = pad_sequences(mark_seq_train, maxlen=max_len_mark, padding='post', truncating='post')
x_train_model = pad_sequences(model_seq_train, maxlen=max_len_model, padding='post', truncating='post')
x_val_mark = pad_sequences(mark_seq_val, maxlen=max_len_mark, padding='post', truncating='post')
x_val_model = pad_sequences(model_seq_val, maxlen=max_len_model, padding='post', truncating='post')
x_test_mark = pad_sequences(mark_seq_test, maxlen=max_len_mark, padding='post', truncating='post')
x_test_model = pad_sequences(model_seq_test, maxlen=max_len_model, padding='post', truncating='post')

# Получаем размеры словарей для Embedding слоев
# Используем num_words из токенизатора + 1 для padding (0 используется для padding)
vocab_size_mark = tokenizer_mark.num_words + 1  # +1 для padding token (0)
vocab_size_model = tokenizer_model.num_words + 1  # +1 для padding token (0)

# Освобождение памяти от промежуточных данных
del mark_seq_train, model_seq_train, mark_seq_val, model_seq_val, mark_seq_test, model_seq_test
print(f"Размер словаря марки: {vocab_size_mark}, максимальная длина: {max_len_mark}")
print(f"Размер словаря модели: {vocab_size_model}, максимальная длина: {max_len_model}")

# Для нормализации данных используются готовые инструменты
y_scaler = StandardScaler()
x_scaler = StandardScaler()

# Нормализация выходных и входных данных по стандартному нормальному распределению
# Сначала обучаем скалеры только на train данных
y_scaler.fit(y_train)
x_scaler.fit(x_train)

# Затем применяем преобразование ко всем выборкам
y_train_scaled = y_scaler.transform(y_train)
y_val_scaled = y_scaler.transform(y_val)
y_test_scaled = y_scaler.transform(y_test)
x_train_scaled = x_scaler.transform(x_train)
x_val_scaled = x_scaler.transform(x_val)
x_test_scaled = x_scaler.transform(x_test)

# Кастомная метрика для MAE в оригинальных единицах
class MAEInverseTransform(Metric):
    def __init__(self, y_scaler, name='mae_inverse', **kwargs):
        super(MAEInverseTransform, self).__init__(name=name, **kwargs)
        self.y_scaler = y_scaler
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Сохраняем масштабирующие параметры
        mean_ = self.y_scaler.mean_[0]
        scale_ = self.y_scaler.scale_[0]
        
        # Преобразуем предсказания и истинные значения обратно в исходный масштаб
        # Используем формулу обратного преобразования напрямую
        y_true_original = y_true * scale_ + mean_
        y_pred_original = y_pred * scale_ + mean_
        
        # Вычисляем MAE в исходных единицах
        # Используем TensorFlow операции вместо NumPy
        mae = tf.reduce_mean(tf.abs(y_true_original - y_pred_original))
        self.total.assign_add(mae)
        self.count.assign_add(1)

    def result(self):
        return self.total / (self.count + 1e-8)

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0)

input1 = Input((x_train.shape[1],))
input2 = Input((x_train_mark.shape[1],))
input3 = Input((x_train_model.shape[1],))

# Первый вход для числовых данных
x1 = input1
x1 = Dense(15, activation="relu")(x1)
x1 = Dense(100, activation="relu")(x1)
x1 = BatchNormalization()(x1)

# Второй вход для данных о марке авто
x2 = input2
x2 = Embedding(input_dim=vocab_size_mark, output_dim=12, input_length=max_len_mark)(x2)
x2 = GlobalAveragePooling1D()(x2)
x2 = Dense(50, activation="relu")(x2)
x2 = Dropout(0.2)(x2)

# Третий вход для данных о модели авто
x3 = input3
x3 = Embedding(input_dim=vocab_size_model, output_dim=16, input_length=max_len_model)(x3)  # немного больше для модели
x3 = GlobalAveragePooling1D()(x3)
x3 = Dense(50, activation="relu")(x3)
x3 = Dropout(0.2)(x3)

# Объединение трех веток
x = concatenate([x1, x2, x3])

# Промежуточный слой
x = Dense(35, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

# Финальный регрессирующий нейрон
x = Dense(1, activation='linear')(x)

# В Model передаются входы и выход
model = Model((input1, input2, input3), x)

mae_inv_metric = MAEInverseTransform(y_scaler)
model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse', metrics=['mse', 'mae', mae_inv_metric])

checkpoint = ModelCheckpoint('best_model.keras', monitor='val_mse', save_best_only=True, verbose=1, mode='min')
early_stop = EarlyStopping(monitor='val_mse', patience=30, restore_best_weights=True, verbose=1, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_mse', factor=0.4, patience=10, min_lr=1e-7, verbose=1, mode='min')

# Обучение модели на обучающем наборе с валидацией на val
history = model.fit(
    [x_train_scaled, x_train_mark, x_train_model],
    y_train_scaled,
    batch_size=64,
    epochs=450,
    callbacks=[checkpoint, early_stop, reduce_lr],
    validation_data=(
        [x_val_scaled, x_val_mark, x_val_model],
        y_val_scaled
    ),
    verbose=1
)

plt.figure(figsize=(10, 6))
plt.plot(history.history['mse'], label='Среднеквадратичная ошибка на обучающем наборе')
plt.plot(history.history['val_mse'], label='Среднеквадратичная ошибка на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Средняя абсолютная ошибка')
plt.legend()
plt.title('Динамика обучения модели')
plt.grid(True)
plt.savefig('plots/loss_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# Оценка модели на тестовом наборе
print("\n=== Оценка на тестовом наборе ===")
pred_test = model.predict([
    x_test_scaled,
    x_test_mark,
    x_test_model
])

# Преобразование предсказаний обратно в исходный масштаб
pred_test = y_scaler.inverse_transform(pred_test)

# Расчет и вывод метрик на тестовом наборе
mse = mean_squared_error(pred_test, y_test)
mae = mean_absolute_error(pred_test, y_test)
mape = mean_absolute_percentage_error(y_test, pred_test)
mean_price = np.mean(y_test)
mse_percent = (np.sqrt(mse) / mean_price) * 100
mae_percent = (mae / mean_price) * 100

print(f'Среднеквадратичная ошибка (MSE): {mse:.2f}')
print(f'Средняя абсолютная ошибка (MAE): {mae:.2f}')
print(f'Средняя абсолютная процентная ошибка (MAPE): {mape*100:.2f}%')
print(f'Средняя цена: {mean_price:.2f}')
print(f'Ошибка MSE в процентах: {mse_percent:.2f}%')
print(f'Ошибка MAE в процентах: {mae_percent:.2f}%')

# Вывод первых 10 предсказаний
for i in range(10):
    print(f'Реальное значение: {y_test[i, 0]:6.2f}  Предсказанное значение: {pred_test[i, 0]:6.2f}  Разница: {abs(y_test[i, 0] - pred_test[i, 0]):6.2f}')

# Визуализация результатов на тестовом наборе
fig, ax = plt.subplots(figsize=(6, 6))
# Плоские массивы для корректного scatter
dtrue_vals = y_test.ravel()
dpred_vals = pred_test.ravel()
ax.scatter(dtrue_vals, dpred_vals)
# Динамические пределы осей, чтобы точки были видны
pad_x = (dtrue_vals.max() - dtrue_vals.min()) * 0.05
pad_y = (dpred_vals.max() - dpred_vals.min()) * 0.05
ax.set_xlim(dtrue_vals.min() - pad_x, dtrue_vals.max() + pad_x)
ax.set_ylim(dpred_vals.min() - pad_y, dpred_vals.max() + pad_y)
ax.plot(plt.xlim(), plt.ylim(), 'r')
plt.xlabel('Правильные значения')
plt.ylabel('Предсказания')
plt.grid()
plt.title('Факт vs Предсказание на тестовом наборе')
plt.savefig('plots/prediction_scatter.png', dpi=300, bbox_inches='tight')
plt.show()