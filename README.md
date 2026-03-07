# ASL Sign Language Recognition

Розпізнавання американської мови жестів (ASL) за допомогою трьох підходів: SimpleCNN, MobileNetV2, MediaPipe + Dense/RandomForest.

## Встановлення

```bash
pip install -r requirements.txt
```

## Датасет

Завантажити [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) та розпакувати в `data/asl_alphabet/`.

Або вказати свій шлях:
```bash
set DATA_DIR=C:\path\to\asl_alphabet
```

## Команди

### Дослідження датасету
```bash
python src/main.py explore
```

### Тренування
```bash
python src/main.py train-cnn
python src/main.py train-mobilenet
python src/main.py train-mediapipe
python src/main.py train-all
```

### Порівняння моделей
```bash
python src/main.py compare
```

### Демо з веб-камерою
```bash
python src/main.py demo --model-path models/mobilenet_best.h5 --model-type mobilenet
python src/main.py demo --model-path models/simple_cnn_best.h5 --model-type cnn
python src/main.py demo --model-path models/mediapipe_dense_best.h5 --model-type landmark_dense
python src/main.py demo --model-path models/random_forest.pkl --model-type random_forest
```

Додаткові параметри демо:
```
--camera 0        індекс камери (за замовчуванням 0)
--smoothing 10    вікно згладжування передбачень
--threshold 0.8   поріг впевненості
```

## Результати

| Модель | Val Accuracy |
|---|---|
| SimpleCNN | ~99% |
| MobileNetV2 | ~99% |
| MediaPipe Dense | ~99% |
| Random Forest | ~97% |

## Структура

```
src/
  main.py                  точка входу
  train.py                 логіка тренування
  evaluate.py              оцінка моделей
  data_loader.py           завантаження даних
  demo.py                  демо з камерою
  models/
    simple_cnn.py
    mobilenet.py
    mediapipe_classifier.py
models/                    збережені ваги
results/                   метрики, графіки, логи
notebooks/                 Jupyter ноутбуки для Kaggle
```
