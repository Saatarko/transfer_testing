import argparse
import logging
import os

import mlflow
import torch
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from torchvision import transforms, datasets
from torchvision.datasets import OxfordIIITPet, ImageFolder
from torch.utils.data import DataLoader, Subset, random_split
from collections import Counter

from utils import get_project_paths
from task_registry import main, task

mlflow.set_tracking_uri('http://localhost:5000')

@task('data:generate_mnist_data')
def generate_mnist_data():
    path = get_project_paths()

    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_mask = train_labels < 5
    test_mask = test_labels < 5

    train_images = train_images[train_mask]
    train_labels = train_labels[train_mask]

    test_images = test_images[test_mask]
    test_labels = test_labels[test_mask]

    (train_images, val_images,
     train_labels, val_labels) = train_test_split(
        train_images,
        train_labels,
        test_size=0.3,
        random_state=42,
        stratify=train_labels
    )

    train_images = train_images.reshape(train_images.shape[0], 28 * 28) / 255.0
    test_images = test_images.reshape(test_images.shape[0], 28 * 28) / 255.0
    val_images = val_images.reshape(val_images.shape[0], 28 * 28) / 255.0

    np.savez_compressed(path['processed_dir']/"dataset_mnist.npz",
                        train_images=train_images,
                        val_images=val_images,
                        test_images=test_images,
                        train_labels=train_labels,
                        val_labels=val_labels,
                        test_labels=test_labels
                        )




@task('data:generate_fash_mnist_data')
def generate_fash_mnist_data():
    path = get_project_paths()

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    (train_images, val_images,
     train_labels, val_labels) = train_test_split(
        train_images,
        train_labels,
        test_size=0.3,
        random_state=42,
        stratify=train_labels
    )

    train_images = train_images.reshape(train_images.shape[0], 28 * 28) / 255.0
    test_images = test_images.reshape(test_images.shape[0], 28 * 28) / 255.0
    val_images = val_images.reshape(val_images.shape[0], 28 * 28) / 255.0

    np.savez_compressed(path['processed_dir']/"dataset_fashion_mnist.npz",
                        train_fashion_images=train_images,
                        val_fashion_images=val_images,
                        test_fashion_images=test_images,
                        train_fashion_labels=train_labels,
                        val_fashion_labels=val_labels,
                        test_fashion_labels=test_labels
                        )

@task('data:download_and_prepare_oxford_pets')
def download_and_prepare_oxford_pets():
    path = get_project_paths()

    # Трансформ
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Средние и стандартные отклонения для ImageNet
    ])

    # Загрузка
    dataset = OxfordIIITPet(
        root=path['raw_dir'], split="trainval", target_types="category",
        transform=transform, download=True
    )

    # Настройка логирования
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Получаем метки всех объектов
    labels = [label for _, label in dataset]

    # Логирование примера меток
    logging.info(f"Пример меток: {labels[:10]}")  # Выводим первые 10 меток для проверки

    # Получаем уникальные классы
    unique_classes = list(set(labels))
    logging.info(f"Количество уникальных классов: {len(unique_classes)}")

    # Разделение на train/val/test
    n_total = len(dataset)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    n_test = n_total - n_train - n_val
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])

    # Функция: сбор всех тензоров в массив
    def collect_data(dataset):
        images, labels = [], []
        for img, label in dataset:
            images.append(img.numpy())
            labels.append(label)
        return np.stack(images), np.array(labels)

    train_images, train_labels = collect_data(train_set)
    val_images, val_labels = collect_data(val_set)
    test_images, test_labels = collect_data(test_set)

    # Сохранение
    np.savez_compressed(
        path['processed_dir'] / "oxford.npz",
        train_images=train_images, train_labels=train_labels,
        val_images=val_images, val_labels=val_labels,
        test_images=test_images, test_labels=test_labels
    )


@task('data:download_and_prepare_imagenette')
def download_and_prepare_imagenette():
    from torchvision.datasets import Imagenette
    import torch
    from torch.utils.data import random_split
    from torchvision import transforms
    import numpy as np

    path = get_project_paths()

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # # Гарантируем, что путь существует
    # path["raw_dir"].mkdir(parents=True, exist_ok=True)
    # path["processed_dir"].mkdir(parents=True, exist_ok=True)

    # Используем Imagenette напрямую
    train_data = Imagenette(root=path["raw_dir"], split="train", download=True, transform=transform)
    val_data = Imagenette(root=path["raw_dir"], split="val", download=True, transform=transform)

    # Делим train на train/test
    generator = torch.Generator().manual_seed(42)
    total_len = len(train_data)
    train_size = int(0.9 * total_len)
    test_size = total_len - train_size
    train_dataset, test_dataset = random_split(train_data, [train_size, test_size], generator=generator)

    def collect_data(dataset):
        images, labels = [], []
        for img, label in dataset:
            images.append(img.numpy())
            labels.append(label)
        return np.stack(images), np.array(labels)

    train_images, train_labels = collect_data(train_dataset)
    test_images, test_labels = collect_data(test_dataset)
    val_images, val_labels = collect_data(val_data)

    np.savez_compressed(
        path["processed_dir"] / "imagenette.npz",
        train_images=train_images, train_labels=train_labels,
        val_images=val_images, val_labels=val_labels,
        test_images=test_images, test_labels=test_labels
    )



@task('data:download_and_prepare_tiny_imagenet')
def download_and_prepare_tiny_imagenet():
    path = get_project_paths()


    # Трансформ
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Средние и стандартные отклонения для ImageNet
    ])

    # Загрузка Tiny ImageNet (пути будут настроены в соответствие с расположением данных)
    dataset_dir = path['raw_dir'] / 'tiny-imagenet-200'

    # Используем ImageFolder, чтобы загрузить изображение по меткам
    dataset = ImageFolder(root=dataset_dir, transform=transform)

    # Логирование
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Фильтрация только 20 классов
    classes_to_keep = dataset.classes[:20]
    class_to_idx = {k: i for i, k in enumerate(classes_to_keep)}

    # Применяем фильтрацию
    filtered_images = []
    filtered_labels = []

    for img, label in dataset:
        if dataset.classes[label] in classes_to_keep:
            filtered_images.append(img)
            filtered_labels.append(class_to_idx[dataset.classes[label]])

    # Преобразуем список изображений и меток в тензоры
    images = torch.stack(filtered_images)
    labels = torch.tensor(filtered_labels)

    # Разделение на train/val/test
    n_total = len(images)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    n_test = n_total - n_train - n_val
    train_set, val_set, test_set = random_split(list(zip(images, labels)), [n_train, n_val, n_test])

    # Функция: сбор всех тензоров в массив
    def collect_data(dataset):
        images, labels = [], []
        for img, label in dataset:
            images.append(img.numpy())
            labels.append(label)
        return np.stack(images), np.array(labels)

    train_images, train_labels = collect_data(train_set)
    val_images, val_labels = collect_data(val_set)
    test_images, test_labels = collect_data(test_set)

    # Сохранение
    np.savez_compressed(
        path['processed_dir'] / "tiny_imagenet_20_classes.npz",
        train_images=train_images, train_labels=train_labels,
        val_images=val_images, val_labels=val_labels,
        test_images=test_images, test_labels=test_labels
    )



@task('data:prepare_animals_dataset')
def prepare_animals_dataset():
    path = get_project_paths()
    raw_data_dir = path["raw_dir"] / "animals"
    processed_path = path["processed_dir"] / "animals.npz"

    # Трансформации
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Можно увеличить при необходимости
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Загружаем датасет из структуры папок
    dataset = ImageFolder(root=raw_data_dir, transform=transform)

    logging.info(f"Найдено {len(dataset)} изображений из {len(dataset.classes)} классов.")
    logging.info(f"Классы: {dataset.classes}")

    # Делим на train/val/test
    n_total = len(dataset)
    n_train = int(0.6 * n_total)
    n_val = int(0.2 * n_total)
    n_test = n_total - n_train - n_val

    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test],
                                                generator=torch.Generator().manual_seed(42))

    def collect_data(ds):
        images, labels = [], []
        for img, label in ds:
            images.append(img.numpy())
            labels.append(label)
        return np.stack(images), np.array(labels)

    # Преобразуем в numpy
    train_images, train_labels = collect_data(train_set)
    val_images, val_labels = collect_data(val_set)
    test_images, test_labels = collect_data(test_set)

    # Сохраняем
    np.savez_compressed(
        processed_path,
        train_images=train_images, train_labels=train_labels,
        val_images=val_images, val_labels=val_labels,
        test_images=test_images, test_labels=test_labels
    )

    logging.info(f"✅ Датасет сохранён в: {processed_path}")







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", help="Список задач для выполнения")
    args = parser.parse_args()

    if args.tasks:
        main(args.tasks)  # Здесь передаем задачи, которые указаны в командной строке
