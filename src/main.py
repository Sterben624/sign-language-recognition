import argparse
import pickle
import sys
import time
from pathlib import Path

import os
ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / 'models'
RESULTS_DIR = ROOT / 'results'
DATA_DIR = Path(os.environ.get('DATA_DIR', str(ROOT / 'data' / 'asl_alphabet')))
CACHE_DIR = RESULTS_DIR / 'landmarks_cache'


def cmd_explore(_args):
    from data_loader import ASLDataLoader, DataConfig
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np

    RESULTS_DIR.mkdir(exist_ok=True)

    loader = ASLDataLoader(DataConfig(data_dir=DATA_DIR))
    df = loader.df

    print(f'Всього зображень : {len(df)}')
    print(f'Класів           : {loader.num_classes}')
    print(f'Класи            : {loader.classes}')

    counts = df['label'].value_counts().sort_index()
    print(f'\nБаланс — min: {counts.min()}, max: {counts.max()}, mean: {counts.mean():.0f}')

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(counts.index, counts.values, color='steelblue')
    ax.set_title('Кількість зображень по класах')
    ax.axhline(counts.mean(), color='red', linestyle='--', label=f'Середнє: {counts.mean():.0f}')
    ax.legend()
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'class_distribution.png', dpi=150)
    plt.close(fig)
    print(f'Збережено: {RESULTS_DIR / "class_distribution.png"}')

    train_df, val_df, test_df = loader.split()
    print(f'\nSplit 70/15/15:')
    print(f'  Train : {len(train_df)}')
    print(f'  Val   : {len(val_df)}')
    print(f'  Test  : {len(test_df)}')

    for name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        split_df.to_csv(RESULTS_DIR / f'{name}_split.csv', index=False)
    print(f'Splits збережено в {RESULTS_DIR}/')


def cmd_train_cnn(_args):
    from data_loader import ASLDataLoader, DataConfig
    from models.simple_cnn import CNNConfig, build_simple_cnn
    from train import ImageModelTrainer, TrainConfig
    from evaluate import ImageModelEvaluator

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    loader = ASLDataLoader(DataConfig(data_dir=DATA_DIR, image_size=(200, 200), batch_size=16))
    train_df, val_df, test_df = loader.split()

    train_gen = loader.image_generator(train_df, augment=True)
    val_gen   = loader.image_generator(val_df,   augment=False)
    test_gen  = loader.image_generator(test_df,  augment=False)

    model = build_simple_cnn(CNNConfig(
        input_shape=(200, 200, 1),
        num_classes=loader.num_classes,
    ))
    model.summary()

    trainer = ImageModelTrainer(model, 'simple_cnn', TrainConfig(
        models_dir=MODELS_DIR, results_dir=RESULTS_DIR, epochs=20, patience=5,
    ))

    t0 = time.time()
    trainer.train(train_gen, val_gen)
    train_time = time.time() - t0
    print(f'\nЧас тренування: {train_time:.1f}s')

    from tensorflow.keras.models import load_model
    best = load_model(MODELS_DIR / 'simple_cnn_best.h5')
    result = ImageModelEvaluator(best, loader.classes, MODELS_DIR / 'simple_cnn_best.h5') \
        .evaluate(test_gen, 'simple_cnn', train_time)

    _print_metrics(result)
    _save_result(result, 'simple_cnn')


def cmd_train_mobilenet(_args):
    from data_loader import ASLDataLoader, DataConfig
    from models.mobilenet import MobileNetConfig
    from train import MobileNetTrainer, TrainConfig
    from evaluate import ImageModelEvaluator

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    loader = ASLDataLoader(DataConfig(data_dir=DATA_DIR, image_size=(200, 200), batch_size=64))
    train_df, val_df, test_df = loader.split()

    train_gen = loader.image_generator(train_df, augment=True)
    val_gen   = loader.image_generator(val_df,   augment=False)
    test_gen  = loader.image_generator(test_df,  augment=False)

    trainer = MobileNetTrainer(
        MobileNetConfig(input_shape=(200, 200, 1), num_classes=loader.num_classes),
        TrainConfig(models_dir=MODELS_DIR, results_dir=RESULTS_DIR,
                    epochs=15, finetune_epochs=10, patience=5),
    )
    trainer.model.summary()

    t0 = time.time()
    trainer.train_with_finetuning(train_gen, val_gen)
    train_time = time.time() - t0
    print(f'\nЧас тренування: {train_time:.1f}s')

    from tensorflow.keras.models import load_model
    best = load_model(MODELS_DIR / 'mobilenet_best.h5')
    result = ImageModelEvaluator(best, loader.classes, MODELS_DIR / 'mobilenet_best.h5') \
        .evaluate(test_gen, 'mobilenet', train_time)

    _print_metrics(result)
    _save_result(result, 'mobilenet')


def cmd_train_mediapipe(_args):
    from data_loader import ASLDataLoader, DataConfig
    from models.mediapipe_classifier import MediaPipeConfig, LandmarkExtractor
    from train import LandmarkModelTrainer, RandomForestTrainer, TrainConfig, extract_and_cache_landmarks
    from evaluate import LandmarkModelEvaluator, RandomForestEvaluator

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    loader = ASLDataLoader(DataConfig(data_dir=DATA_DIR))
    train_df, val_df, test_df = loader.split()

    mediapipe_config = MediaPipeConfig(num_classes=loader.num_classes)
    train_config = TrainConfig(models_dir=MODELS_DIR, results_dir=RESULTS_DIR,
                               epochs=30, patience=7)

    print('Витягування landmarks (кешується після першого запуску)...')
    with LandmarkExtractor(mediapipe_config) as extractor:
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = extract_and_cache_landmarks(
            loader, extractor, (train_df, val_df, test_df), CACHE_DIR
        )

    print(f'Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}')

    print('\n── Dense мережа ──')
    dense_trainer = LandmarkModelTrainer(train_config, mediapipe_config)
    t0 = time.time()
    dense_trainer.train(X_train, y_train, X_val, y_val)
    dense_time = time.time() - t0
    dense_trainer.save()

    from tensorflow.keras.models import load_model
    dense_model = load_model(MODELS_DIR / 'mediapipe_dense_best.h5')
    dense_result = LandmarkModelEvaluator(
        dense_model, loader.classes, MODELS_DIR / 'mediapipe_dense_best.h5'
    ).evaluate(X_test, y_test, 'mediapipe_dense', dense_time)
    _print_metrics(dense_result)
    _save_result(dense_result, 'dense')

    print('\n── Random Forest ──')
    rf_trainer = RandomForestTrainer(train_config, mediapipe_config)
    t0 = time.time()
    rf_trainer.train(X_train, y_train)
    rf_time = time.time() - t0
    rf_trainer.save()

    rf_result = RandomForestEvaluator(
        MODELS_DIR / 'random_forest.pkl', loader.classes
    ).evaluate(X_test, y_test, rf_time)
    _print_metrics(rf_result)
    _save_result(rf_result, 'rf')


def cmd_train_all(args):
    cmd_train_cnn(args)
    cmd_train_mobilenet(args)
    cmd_train_mediapipe(args)
    cmd_compare(args)


def cmd_compare(_args):
    from evaluate import ComparisonReport

    result_files = {
        'simple_cnn': RESULTS_DIR / 'simple_cnn_result.pkl',
        'mobilenet':  RESULTS_DIR / 'mobilenet_result.pkl',
        'dense':      RESULTS_DIR / 'dense_result.pkl',
        'rf':         RESULTS_DIR / 'rf_result.pkl',
    }

    results = []
    for name, path in result_files.items():
        if not path.exists():
            print(f'Пропущено {name} — файл не знайдено: {path}')
            continue
        with open(path, 'rb') as f:
            results.append(pickle.load(f))

    if not results:
        print('Немає збережених результатів. Спочатку запусти train.')
        return

    report = ComparisonReport(results, RESULTS_DIR)
    df = report.save_metrics_table()
    report.plot_confusion_matrices()
    report.plot_metrics_comparison()
    report.plot_training_history({
        'simple_cnn':      RESULTS_DIR / 'simple_cnn_history.csv',
        'mobilenet':       RESULTS_DIR / 'mobilenet_history.csv',
        'mediapipe_dense': RESULTS_DIR / 'mediapipe_dense_history.csv',
    })

    print('\n── Порівняння моделей ──')
    print(df.to_string(index=False))
    print(f'\nГрафіки збережено в {RESULTS_DIR}/')


def cmd_demo(args):
    from demo import DemoConfig, WebcamDemo

    CLASSES = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['del', 'nothing', 'space']

    WebcamDemo(DemoConfig(
        model_path=Path(args.model_path),
        class_names=CLASSES,
        model_type=args.model_type,
        smoothing_window=args.smoothing,
        confidence_threshold=args.threshold,
        camera_index=args.camera,
    )).run()


def _print_metrics(result) -> None:
    print(f'\n  Accuracy       : {result.accuracy:.4f}')
    print(f'  Covered acc    : {result.covered_accuracy:.4f}  (threshold={0.8}, coverage={result.coverage:.2%})')
    print(f'  Macro F1       : {result.macro_f1:.4f}')
    print(f'  Inference      : {result.inference_ms_per_sample:.3f} ms/sample')
    print(f'  Розмір         : {result.model_size_mb:.2f} MB')
    print(f'  Час трен.      : {result.train_time_sec:.1f}s')


def _save_result(result, name: str) -> None:
    path = RESULTS_DIR / f'{name}_result.pkl'
    with open(path, 'wb') as f:
        pickle.dump(result, f)
    print(f'  Збережено : {path}')


def main():
    parser = argparse.ArgumentParser(
        prog='main.py',
        description='ASL Sign Language Recognition',
    )
    sub = parser.add_subparsers(dest='command', required=True)

    sub.add_parser('explore',          help='Дослідження датасету')
    sub.add_parser('train-cnn',        help='Тренування Simple CNN')
    sub.add_parser('train-mobilenet',  help='Тренування MobileNetV2')
    sub.add_parser('train-mediapipe',  help='Тренування MediaPipe + Dense/RF')
    sub.add_parser('train-all',        help='Тренування всіх моделей')
    sub.add_parser('compare',          help='Порівняння збережених результатів')

    demo_p = sub.add_parser('demo', help='Демо з веб-камерою')
    demo_p.add_argument('--model-path', required=True)
    demo_p.add_argument('--model-type', required=True,
                        choices=['cnn', 'mobilenet', 'landmark_dense', 'random_forest'])
    demo_p.add_argument('--camera',    type=int,   default=0)
    demo_p.add_argument('--smoothing', type=int,   default=10)
    demo_p.add_argument('--threshold', type=float, default=0.8)

    args = parser.parse_args()

    commands = {
        'explore':         cmd_explore,
        'train-cnn':       cmd_train_cnn,
        'train-mobilenet': cmd_train_mobilenet,
        'train-mediapipe': cmd_train_mediapipe,
        'train-all':       cmd_train_all,
        'compare':         cmd_compare,
        'demo':            cmd_demo,
    }
    commands[args.command](args)


if __name__ == '__main__':
    main()
