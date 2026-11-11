# Анифрод, классификация

github не готов принять файл больше 100 мб

Ссылка на датасет: https://www.kaggle.com/datasets/kartik2112/fraud-detection/data
## Постановка задачи
Нужно определить, мошенническая транзакция или нет, таргет is_fraud. Данные в двух файлах, 1.3 млн транзакций на обучение и 556 тысяч на тест

Признаки:
- trans_date_trans_time - дата и время транзакции
- merchant - продавец
- category - категория покупки
- amt - сумма
- gender, dob, job - пол, дата рождения и профессия клиента
- first, last, street, city, state, zip - имя и адрес клиента
- lat, long, merch_lat, merch_long - координаты клиента и продавца
- city_pop - население города
- cc_num, trans_num - номер карты и транзакции

## EDA
Пропусков нет. Мошеннических транзакций меньше 1%, сильный дисбаланс классов

В amt и city_pop много выбросов. Мошеннические транзакции разбросаны по всей территории, поэтому признак региона делать не стал

Имя, фамилия, улица и город почти уникальны для каждого клиента, модель начнет запоминать людей, поэтому эти признаки убрал

Gender закодировал через OneHot, merchant, job и state через TargetEncoder, category через частоту

## Baseline
Сравнил DummyClassifier и LightGBM без настройки:
- Dummy: F1 0
- LightGBM: F1 0.647

Сравнивал по F1

## Improvements
Попробовал SMOTENC, чтобы добавить мошеннических транзакций. Стало хуже, F1 упал до 0.52, синтетические примеры создали шум

Потом заменил энкодеры на встроенную обработку категорий в LightGBM. F1 получился 0.54, precision вырос почти до 1, но recall сильно упал. В итоге оставил категории без энкодеров, а recall поднимал подбором порога

## Гиперпараметры
Подбирал параметры через Optuna по F1, по 30 триалов на модель, потом подобрал порог через TunedThresholdClassifierCV

Лучшие параметры:
- RandomForest: max_depth 29, max_features 0.16
- LightGBM: learning_rate 0.15, n_estimators 1000, num_leaves 135, min_child_samples 700
- XGBoost: learning_rate 0.057, n_estimators 900, max_depth 3
- CatBoost: learning_rate 0.036, depth 10, l2_leaf_reg 10

Результаты на тесте:
- RandomForest: F1 0.704
- LightGBM: F1 0.643
- CatBoost: F1 0.598
- XGBoost: F1 0.232

В итоговый файл сохранил LightGBM и CatBoost, у них лучше всего F1 на кросс валидации

## Файлы
- exploration_data_analysis.ipynb - EDA
- base_line.ipynb - baseline
- improvments.ipynb - SMOTE и категории
- hptunning.ipynb - подбор гиперпараметров и порога
- funcs.py - функции предобработки
- final_result.joblib - итоговые модели
