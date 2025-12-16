import pandas as pd
cars = pd.read_csv('japan_cars_dataset.csv', sep=',')

# Удалим строки с пустыми значениями
cars = cars.dropna()

import numpy as np
from keras import utils
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Input, concatenate
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
# Токенизатор
from tensorflow.keras.preprocessing.text import Tokenizer

# Удаляем первую колонку без имени
cars = cars.drop(columns=cars.columns[0])

# Удаляем редкие марки (<3 вхождений) и считаем, сколько строк ушло
rows_before = len(cars)
rare_marks = cars['mark'].value_counts()[lambda s: s < 3].index
cars = cars[~cars['mark'].isin(rare_marks)]
removed_rows = rows_before - len(cars)
print(f"Удалено строк с редкими марками (<3): {removed_rows}")


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
numeric_columns = ['price', 'year', 'mileage', 'engine_capacity']

# Удаляем выбросы
cars = remove_outliers_iqr(cars, numeric_columns, multiplier=1.2)

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
    num_words=3000,                                          # объем словаря
    filters='!"«»#$№%&()*+,-–—./:;<=>?@[\\]^_`{|}~\t\n\xa0', # убираемые из текста ненужные символы
    lower=True,                                              # приведение слов к нижнему регистру
    split=' ',                                               # разделитель слов
    oov_token='unknown',                                     # токен для слов, которые не вошли в словарь
    char_level=False                                         # разделяем по словам, а не по единичным символам
)

tokenizer_model = Tokenizer(
    num_words=3000,                                          # объем словаря
    filters='!"«»#$№%&()*+,-–—./:;<=>?@[\\]^_`{|}~\t\n\xa0', # убираемые из текста ненужные символы
    lower=True,                                              # приведение слов к нижнему регистру
    split=' ',                                               # разделитель слов
    oov_token='unknown',                                     # токен для слов, которые не вошли в словарь
    char_level=False                                         # разделяем по словам, а не по единичным символам
)

# Построение частотного словаря по текстам образования
# tokenizer.fit_on_texts(cars['mark_model'])
tokenizer_mark.fit_on_texts(cars['mark'])
tokenizer_model.fit_on_texts(cars['model'])
# Преобразование текстов в последовательность индексов согласно частотному словарю
mark_seq = tokenizer_mark.texts_to_sequences(cars['mark'])
model_seq = tokenizer_model.texts_to_sequences(cars['model'])
# Преобразование последовательностей индексов в bag of words
x_train_mark = tokenizer_mark.sequences_to_matrix(mark_seq)
x_train_model = tokenizer_model.sequences_to_matrix(model_seq)


# Освобождение памяти от промежуточных данных
# del mark_model_seq, tokenizer
del mark_seq, model_seq, tokenizer_mark, tokenizer_model
print("done")

# Удалены классы интервалов, так как признаки теперь числовые

# transmission_class  = {'cvt': 0,
#                        'mt': 1,
#                        'at': 2
transmission_class  = {'mt': 0,
                       'at': 1
                       }
drive_class         = {'2wd': 0,
                       '4wd': 1
                       }
hand_drive_class    = {'lhd': 0,
                       'rhd': 1
                       }
# fuel_class          = {'cng': 0,
#                        'hybrid': 1,
#                        'lpg': 2,
#                        'diesel': 3,
#                        'gasoline': 4
#                       }
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

# Формирование выборки из загруженного набора данных
x_train, y_train = get_train_data(cars)

# Для нормализации данных используются готовые инструменты
y_scaler = StandardScaler()
x_scaler = StandardScaler()

# Нормализация выходных и входных данных по стандартному нормальному распределению
y_train_scaled = y_scaler.fit_transform(y_train)
x_train_scaled = x_scaler.fit_transform(x_train)

input1 = Input((x_train.shape[1],))
input2 = Input((x_train_mark.shape[1],))
input3 = Input((x_train_model.shape[1],))

# Первый вход для числовых данных
x1 = input1
x1 = Dense(20, activation="relu")(x1)
x1 = Dense(500, activation="relu")(x1)
x1 = Dense(200, activation="relu")(x1)


# Второй вход для данных о марке авто
x2 = input2
x2 = Dense(20, activation="relu")(x2)
x2 = Dense(200, activation="relu")(x2)
x2 = Dropout(0.3)(x2)

# Третий вход для данных о модели авто
x3 = input3
x3 = Dense(20, activation="relu")(x3)
x3 = Dense(200, activation="relu")(x3)
x3 = Dropout(0.3)(x3)

# Объединение четырех веток
x = concatenate([x1, x2, x3])

# Промежуточный слой
x = Dense(30, activation='relu')(x)
x = Dropout(0.5)(x)

# Финальный регрессирующий нейрон
x = Dense(1, activation='linear')(x)

# В Model передаются входы и выход
model = Model((input1, input2, input3), x)

model.compile(optimizer=Adam(learning_rate=1e-5), loss='mae', metrics=['mae'])

checkpoint = ModelCheckpoint('best_model.keras', monitor='val_mae', save_best_only=True, verbose=1, mode='min')
early_stop = EarlyStopping(monitor='val_mae', patience=30, restore_best_weights=True, verbose=1, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_mae', factor=0.4, patience=10, min_lr=1e-7, verbose=1, mode='min')

# Определяем размер разделения (85% на обучение, 15% на валидацию - больше данных для обучения)
split_idx = int(len(x_train) * 0.85)

history = model.fit([x_train_scaled[:split_idx], x_train_mark[:split_idx], x_train_model[:split_idx]],
                    y_train_scaled[:split_idx],
                    batch_size=32,
                    epochs=450,
                    callbacks=[checkpoint, early_stop, reduce_lr],
                    validation_data=([x_train_scaled[split_idx:], x_train_mark[split_idx:], x_train_model[split_idx:]], y_train_scaled[split_idx:]),
                    verbose=1)

plt.plot(history.history['mae'], label='Средняя абсолютная ошибка на обучающем наборе')
plt.plot(history.history['val_mae'], label='Средняя абсолютная ошибка на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Средняя абсолютная ошибка')
plt.legend()
# plt.show()

pred = model.predict([x_train_scaled[split_idx:], x_train_mark[split_idx:], x_train_model[split_idx:]])

pred = y_scaler.inverse_transform(pred)    # Обратная нормированию процедура

#print('Средняя абсолютная ошибка:', mean_absolute_error(pred, y_train[split_idx:]), '\n') # расчет средней абсолютной ошибки
# Расчет и вывод ошибок
mse = mean_squared_error(pred, y_train[split_idx:])
mae = mean_absolute_error(pred, y_train[split_idx:])
mean_price = np.mean(y_train[split_idx:])
mse_percent = (np.sqrt(mse) / mean_price) * 100
mae_percent = (mae / mean_price) * 100

print(f'Среднеквадратичная ошибка (MSE): {mse:.2f}')
print(f'Средняя абсолютная ошибка (MAE): {mae:.2f}')
print(f'Средняя цена: {mean_price:.2f}')
print(f'Ошибка MSE в процентах: {mse_percent:.2f}%')
print(f'Ошибка MAE в процентах: {mae_percent:.2f}%')

for i in range(10):
    print('Реальное значение: {:6.2f}  Предсказанное значение: {:6.2f}  Разница: {:6.2f}'.format(y_train[split_idx:][i, 0],
                                                                                                 pred[i, 0],
                                                                                                 abs(y_train[split_idx:][i, 0] - pred[i, 0])))
fig, ax = plt.subplots(figsize=(6, 6))
# Плоские массивы для корректного scatter
true_vals = y_train[split_idx:].ravel()
pred_vals = pred.ravel()
ax.scatter(true_vals, pred_vals)          # Отрисовка точечного графика
# Динамические пределы осей, чтобы точки были видны
pad_x = (true_vals.max() - true_vals.min()) * 0.05
pad_y = (pred_vals.max() - pred_vals.min()) * 0.05
ax.set_xlim(true_vals.min() - pad_x, true_vals.max() + pad_x)
ax.set_ylim(pred_vals.min() - pad_y, pred_vals.max() + pad_y)
ax.plot(plt.xlim(), plt.ylim(), 'r')          # Отрисовка диагональной линии
plt.xlabel('Правильные значения')
plt.ylabel('Предсказания')
plt.grid()
# plt.show()