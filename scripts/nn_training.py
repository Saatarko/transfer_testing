import argparse

import mlflow
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset
import mlflow
from torchvision import datasets
from torchvision import models
from torchvision import transforms
from torchvision.models import resnet18
from utils import get_project_paths, log_confusion_matrix
from task_registry import main, task


mlflow.set_tracking_uri('http://localhost:5000')

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.input = nn.Linear(784, 128)
        self.hidden = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.Tanh()
        self.output = nn.Linear(64, 5)  # 5 классов

    def forward(self, x):
        x = self.input(x)
        x = self.activation(x)
        x = self.hidden(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output(x)  # logits — без softmax
        return x



@task('data:train_nn')
def train_nn():
    path = get_project_paths()

    data = np.load(path['processed_dir']/"dataset_mnist.npz")
    X_train = torch.tensor(data["train_images"], dtype=torch.float32)
    y_train = torch.tensor(data["train_labels"], dtype=torch.long)
    X_val = torch.tensor(data["val_images"], dtype=torch.float32)
    y_val = torch.tensor(data["val_labels"], dtype=torch.long)



    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)


    # === Обучение ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 50

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    with mlflow.start_run():
        for epoch in range(num_epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_train_loss = running_loss / total
            epoch_train_acc = correct / total
            train_losses.append(epoch_train_loss)
            train_accuracies.append(epoch_train_acc)

            # Валидация
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            epoch_val_loss = val_loss / val_total
            epoch_val_acc = val_correct / val_total
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_acc)

            # Логирование метрик в MLflow
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_train_acc, step=epoch)
            mlflow.log_metric("val_accuracy", epoch_val_acc, step=epoch)

            print(f"[{epoch + 1}/{num_epochs}] Train Loss: {epoch_train_loss:.4f}, "
                  f"Val Loss: {epoch_val_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
                  f"Val Acc: {epoch_val_acc:.4f}")

        # === Сохраняем модель ===
        model_path = path['models_dir']/"mnist_nn_model.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)

        # === Графики ===
        epochs = range(1, num_epochs + 1)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss")

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, label="Train Accuracy")
        plt.plot(epochs, val_accuracies, label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy")

        plt.tight_layout()
        plot_path = path['image_dir']/"mnist_training_metrics.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()

        return model



@task('data:evaluate_model_on_test')
def evaluate_model_on_test():
    path = get_project_paths()

    model_path = path['models_dir'] / "mnist_nn_model.pth"

    data = np.load(path['processed_dir'] / "dataset_mnist.npz")

    X_test = torch.tensor(data["test_images"], dtype=torch.float32)
    y_test = torch.tensor(data["test_labels"], dtype=torch.long)

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Воссоздаем модель и загружаем веса
    model = SimpleNN()
    model.load_state_dict(torch.load(model_path))

    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())  # возвращаем на cpu для подсчета
            all_labels.extend(labels.cpu().numpy())  # то же самое

    # Логируем матрицу ошибок
    model_name_tag = 'mnist'
    log_confusion_matrix(all_preds, all_labels, model_name_tag)

    # Вычисляем точность на тестовом наборе
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    return accuracy




@task('data:train_no_frease_nn')
def train_no_frease_nn():
    path = get_project_paths()

    data = np.load(path['processed_dir']/"dataset_fashion_mnist.npz")
    X_train = torch.tensor(data["train_fashion_images"], dtype=torch.float32)
    y_train = torch.tensor(data["train_fashion_labels"], dtype=torch.long)
    X_val = torch.tensor(data["val_fashion_images"], dtype=torch.float32)
    y_val = torch.tensor(data["val_fashion_labels"], dtype=torch.long)



    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)


    # === Обучение ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrained_model_path = path['models_dir'] / "mnist_nn_model.pth"
    model = SimpleNN().to(device)
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))

    # Заменяем выходной слой (5 -> 10 классов)
    model.output = nn.Linear(64, 10).to(device)

    # Оптимизатор: обучаем все параметры (без заморозки)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 50

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    with mlflow.start_run():
        for epoch in range(num_epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_train_loss = running_loss / total
            epoch_train_acc = correct / total
            train_losses.append(epoch_train_loss)
            train_accuracies.append(epoch_train_acc)

            # Валидация
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            epoch_val_loss = val_loss / val_total
            epoch_val_acc = val_correct / val_total
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_acc)

            # Логирование метрик в MLflow
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_train_acc, step=epoch)
            mlflow.log_metric("val_accuracy", epoch_val_acc, step=epoch)

            print(f"[{epoch + 1}/{num_epochs}] Train Loss: {epoch_train_loss:.4f}, "
                  f"Val Loss: {epoch_val_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
                  f"Val Acc: {epoch_val_acc:.4f}")

        # === Сохраняем модель ===
        model_path = path['models_dir']/"mnist_fash_no_frease_nn_model.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)

        # === Графики ===
        epochs = range(1, num_epochs + 1)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss")

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, label="Train Accuracy")
        plt.plot(epochs, val_accuracies, label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy")

        plt.tight_layout()
        plot_path = path['image_dir']/"mnist_fash_no_frease_training_metrics.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()

        return model



@task('data:evaluate_fash_no_frease_model_on_test')
def evaluate_fash_no_frease_model_on_test():
    path = get_project_paths()

    model_path = path['models_dir'] / "mnist_fash_no_frease_nn_model.pth"

    data = np.load(path['processed_dir'] / "dataset_fashion_mnist.npz")

    X_test = torch.tensor(data["test_fashion_images"], dtype=torch.float32)
    y_test = torch.tensor(data["test_fashion_labels"], dtype=torch.long)

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Воссоздаем модель и загружаем веса
    model = SimpleNN()
    model.output = nn.Linear(64, 10)  # 10 классов
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())  # возвращаем на cpu для подсчета
            all_labels.extend(labels.cpu().numpy())  # то же самое

    # Логируем матрицу ошибок
    model_name_tag = 'mnist_nofrease'
    log_confusion_matrix(all_preds, all_labels, model_name_tag)

    # Вычисляем точность на тестовом наборе
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    return accuracy




@task('data:train_full_frease_nn')
def train_full_frease_nn():
    path = get_project_paths()

    data = np.load(path['processed_dir']/"dataset_fashion_mnist.npz")
    X_train = torch.tensor(data["train_fashion_images"], dtype=torch.float32)
    y_train = torch.tensor(data["train_fashion_labels"], dtype=torch.long)
    X_val = torch.tensor(data["val_fashion_images"], dtype=torch.float32)
    y_val = torch.tensor(data["val_fashion_labels"], dtype=torch.long)



    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)


    # === Обучение ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrained_model_path = path['models_dir'] / "mnist_nn_model.pth"
    model = SimpleNN().to(device)
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))

    # Заменяем выходной слой (5 -> 10 классов)
    model.output = nn.Linear(64, 10).to(device)

    for param in model.input.parameters():
        param.requires_grad = False
    for param in model.hidden.parameters():
        param.requires_grad = False

    # Оптимизатор: обучаем все параметры (без заморозки)
    optimizer = optim.Adam(model.output.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 50

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    with mlflow.start_run():
        for epoch in range(num_epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_train_loss = running_loss / total
            epoch_train_acc = correct / total
            train_losses.append(epoch_train_loss)
            train_accuracies.append(epoch_train_acc)

            # Валидация
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            epoch_val_loss = val_loss / val_total
            epoch_val_acc = val_correct / val_total
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_acc)

            # Логирование метрик в MLflow
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_train_acc, step=epoch)
            mlflow.log_metric("val_accuracy", epoch_val_acc, step=epoch)

            print(f"[{epoch + 1}/{num_epochs}] Train Loss: {epoch_train_loss:.4f}, "
                  f"Val Loss: {epoch_val_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
                  f"Val Acc: {epoch_val_acc:.4f}")

        # === Сохраняем модель ===
        model_path = path['models_dir']/"mnist_full_frease_nn_model.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)

        # === Графики ===
        epochs = range(1, num_epochs + 1)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss")

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, label="Train Accuracy")
        plt.plot(epochs, val_accuracies, label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy")

        plt.tight_layout()
        plot_path = path['image_dir']/"mnist_full_frease_training_metrics.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()

        return model



@task('data:evaluate_fash_full_frease_model_on_test')
def evaluate_fash_full_frease_model_on_test():
    path = get_project_paths()

    model_path = path['models_dir'] / "mnist_full_frease_nn_model.pth"

    data = np.load(path['processed_dir'] / "dataset_fashion_mnist.npz")

    X_test = torch.tensor(data["test_fashion_images"], dtype=torch.float32)
    y_test = torch.tensor(data["test_fashion_labels"], dtype=torch.long)

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Воссоздаем модель и загружаем веса
    model = SimpleNN()
    model.output = nn.Linear(64, 10)  # 10 классов
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())  # возвращаем на cpu для подсчета
            all_labels.extend(labels.cpu().numpy())  # то же самое

    # Логируем матрицу ошибок
    model_name_tag = 'mnist_full_frease'
    log_confusion_matrix(all_preds, all_labels, model_name_tag)

    # Вычисляем точность на тестовом наборе
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    return accuracy




@task('data:train_part_frease_nn')
def train_part_frease_nn():
    path = get_project_paths()

    data = np.load(path['processed_dir']/"dataset_fashion_mnist.npz")
    X_train = torch.tensor(data["train_fashion_images"], dtype=torch.float32)
    y_train = torch.tensor(data["train_fashion_labels"], dtype=torch.long)
    X_val = torch.tensor(data["val_fashion_images"], dtype=torch.float32)
    y_val = torch.tensor(data["val_fashion_labels"], dtype=torch.long)



    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)


    # === Обучение ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrained_model_path = path['models_dir'] / "mnist_nn_model.pth"
    model = SimpleNN().to(device)
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))

    # Заменяем выходной слой (5 -> 10 классов)
    model.output = nn.Linear(64, 10).to(device)

    for param in model.input.parameters():
        param.requires_grad = False
    for param in model.hidden.parameters():
        param.requires_grad = False

    # Оптимизатор: обучаем все параметры (без заморозки)
    optimizer = optim.Adam(model.output.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 50

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    unfrozen = False


    with mlflow.start_run():
        for epoch in range(num_epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            mlflow.log_metric("model_unfrozen", 0, step=epoch)
            if not unfrozen and epoch == num_epochs // 2:
                for param in model.parameters():
                    param.requires_grad = True
                optimizer = optim.Adam(model.parameters(), lr=1e-4)
                unfrozen = True
                mlflow.log_metric("model_unfrozen", 1, step=epoch)

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_train_loss = running_loss / total
            epoch_train_acc = correct / total
            train_losses.append(epoch_train_loss)
            train_accuracies.append(epoch_train_acc)

            # Валидация
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            epoch_val_loss = val_loss / val_total
            epoch_val_acc = val_correct / val_total
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_acc)

            # Логирование метрик в MLflow
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_train_acc, step=epoch)
            mlflow.log_metric("val_accuracy", epoch_val_acc, step=epoch)

            print(f"[{epoch + 1}/{num_epochs}] Train Loss: {epoch_train_loss:.4f}, "
                  f"Val Loss: {epoch_val_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
                  f"Val Acc: {epoch_val_acc:.4f}")

        # === Сохраняем модель ===
        model_path = path['models_dir']/"mnist_part_frease_nn_model.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)

        # === Графики ===
        epochs = range(1, num_epochs + 1)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss")

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, label="Train Accuracy")
        plt.plot(epochs, val_accuracies, label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy")

        plt.tight_layout()
        plot_path = path['image_dir']/"mnist_part_frease_training_metrics.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()

        return model



@task('data:evaluate_part_full_frease_model_on_test')
def evaluate_part_full_frease_model_on_test():
    path = get_project_paths()

    model_path = path['models_dir'] / "mnist_part_frease_nn_model.pth"

    data = np.load(path['processed_dir'] / "dataset_fashion_mnist.npz")

    X_test = torch.tensor(data["test_fashion_images"], dtype=torch.float32)
    y_test = torch.tensor(data["test_fashion_labels"], dtype=torch.long)

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Воссоздаем модель и загружаем веса
    model = SimpleNN()
    model.output = nn.Linear(64, 10)  # 10 классов
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())  # возвращаем на cpu для подсчета
            all_labels.extend(labels.cpu().numpy())  # то же самое

    # Логируем матрицу ошибок
    model_name_tag = 'mnist_part_frease'
    log_confusion_matrix(all_preds, all_labels, model_name_tag)

    # Вычисляем точность на тестовом наборе
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    return accuracy



class SimpleCNN(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),

            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.net(x)



# Трансформации для аугментации данных
train_transform = T.Compose([
    T.RandomResizedCrop(64, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
])

val_transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
])



@task('data:train_cnn')
def train_cnn():
    path = get_project_paths()

    # Загрузка данных
    data = np.load(path['processed_dir'] / "oxford.npz")
    X_train = torch.tensor(data["train_images"], dtype=torch.float32)
    y_train = torch.tensor(data["train_labels"], dtype=torch.long)
    X_val = torch.tensor(data["val_images"], dtype=torch.float32)
    y_val = torch.tensor(data["val_labels"], dtype=torch.long)

    # Даталоадеры
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

    # === Обучение ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(dropout=0.5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    num_epochs = 50

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []


    with mlflow.start_run():
        for epoch in range(num_epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_train_loss = running_loss / total
            epoch_train_acc = correct / total
            train_losses.append(epoch_train_loss)
            train_accuracies.append(epoch_train_acc)

            # Валидация
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            epoch_val_loss = val_loss / val_total
            epoch_val_acc = val_correct / val_total
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_acc)


            # Логирование метрик в MLflow
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_train_acc, step=epoch)
            mlflow.log_metric("val_accuracy", epoch_val_acc, step=epoch)

            # Печать прогресса
            print(f"[{epoch + 1}/{num_epochs}] Train Loss: {epoch_train_loss:.4f}, "
                  f"Val Loss: {epoch_val_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
                  f"Val Acc: {epoch_val_acc:.4f}")

            # Обновление scheduler
            scheduler.step(epoch_val_loss)

        # === Сохраняем модель ===
        model_path = path['models_dir'] / "oxford_cnn_model.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)

        # === Графики ===
        epochs = range(1, num_epochs + 1)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss")

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, label="Train Accuracy")
        plt.plot(epochs, val_accuracies, label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy")

        plt.tight_layout()
        plot_path = path['image_dir'] / "oxford_training_metrics.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()

        return model



@task('data:evaluate_cnn_oxford')
def evaluate_cnn_oxford():
    path = get_project_paths()

    model_path = path['models_dir'] / "oxford_cnn_model.pth"

    data = np.load(path['processed_dir'] / "oxford.npz")

    X_test = torch.tensor(data["test_images"], dtype=torch.float32)
    y_test = torch.tensor(data["test_labels"], dtype=torch.long)

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Воссоздаем модель и загружаем веса
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())  # возвращаем на cpu для подсчета
            all_labels.extend(labels.cpu().numpy())  # то же самое

    # Логируем матрицу ошибок
    model_name_tag = 'oxford'
    log_confusion_matrix(all_preds, all_labels, model_name_tag)

    # Вычисляем точность на тестовом наборе
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    return accuracy


@task('data:train_cnn_imagenette_base')
def train_cnn_imagenette_base():
    path = get_project_paths()

    # Трансформ
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # Средние и стандартные отклонения для ImageNet
    ])

    # Загрузка Imagenette
    imagenette_data = datasets.ImageFolder(root=path['raw_dir'] / 'imagenette', transform=transform)
    train_size = int(0.8 * len(imagenette_data))
    val_size = len(imagenette_data) - train_size
    train_dataset, val_dataset = random_split(imagenette_data, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # Использование модели, предобученной на Imagenette
    model = models.resnet18(pretrained=True)

    # Замораживаем все слои, кроме последнего
    for param in model.parameters():
        param.requires_grad = False

    # Меняем последний слой на соответствующий для 10 классов
    model.fc = nn.Linear(model.fc.in_features, 10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    num_epochs = 50
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    with mlflow.start_run():
        for epoch in range(num_epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_train_loss = running_loss / total
            epoch_train_acc = correct / total
            train_losses.append(epoch_train_loss)
            train_accuracies.append(epoch_train_acc)

            # Валидация
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            epoch_val_loss = val_loss / val_total
            epoch_val_acc = val_correct / val_total
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_acc)

            # Логирование метрик в MLflow
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_train_acc, step=epoch)
            mlflow.log_metric("val_accuracy", epoch_val_acc, step=epoch)

            # Печать прогресса
            print(f"[{epoch + 1}/{num_epochs}] Train Loss: {epoch_train_loss:.4f}, "
                  f"Val Loss: {epoch_val_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
                  f"Val Acc: {epoch_val_acc:.4f}")

        # === Сохраняем модель ===
        model_path = path['models_dir'] / "imagenette_base_cnn_model.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)

        # === Графики ===
        epochs = range(1, num_epochs + 1)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss")

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, label="Train Accuracy")
        plt.plot(epochs, val_accuracies, label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy")

        plt.tight_layout()
        plot_path = path['image_dir'] / "imagenette_training_metrics.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()

        return model




@task('data:train_cnn_imagenette')
def train_cnn_imagenette():
    path = get_project_paths()

    # Загрузка данных
    data = np.load(path['processed_dir'] / "imagenette.npz")
    X_train = torch.tensor(data["train_images"], dtype=torch.float32)
    y_train = torch.tensor(data["train_labels"], dtype=torch.long)
    X_val = torch.tensor(data["val_images"], dtype=torch.float32)
    y_val = torch.tensor(data["val_labels"], dtype=torch.long)

    # Даталоадеры
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

    # === Обучение ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(dropout=0.5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    num_epochs = 250

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    patience = 5  # Количество эпох без улучшения, после которых остановим обучение
    best_val_loss = float('inf')
    epochs_since_improvement = 0

    with mlflow.start_run():
        for epoch in range(num_epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_train_loss = running_loss / total
            epoch_train_acc = correct / total
            train_losses.append(epoch_train_loss)
            train_accuracies.append(epoch_train_acc)

            # Проверка на улучшение
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if epochs_since_improvement >= patience:
                print("Early stopping triggered")
                break

            # Валидация
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            epoch_val_loss = val_loss / val_total
            epoch_val_acc = val_correct / val_total
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_acc)


            # Логирование метрик в MLflow
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_train_acc, step=epoch)
            mlflow.log_metric("val_accuracy", epoch_val_acc, step=epoch)

            # Печать прогресса
            print(f"[{epoch + 1}/{num_epochs}] Train Loss: {epoch_train_loss:.4f}, "
                  f"Val Loss: {epoch_val_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
                  f"Val Acc: {epoch_val_acc:.4f}")

            # Обновление scheduler
            scheduler.step(epoch_val_loss)

        # === Сохраняем модель ===
        model_path = path['models_dir'] / "imagenette_cnn_model.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)

        num_epochs_actual = epoch + 1 if epochs_since_improvement < patience else epoch

        # Подготовим графики с реальным количеством эпох
        epochs = range(1, num_epochs_actual + 1)

        # Графики
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses[:num_epochs_actual], label="Train Loss")
        plt.plot(epochs, val_losses[:num_epochs_actual], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss")

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies[:num_epochs_actual], label="Train Accuracy")
        plt.plot(epochs, val_accuracies[:num_epochs_actual], label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy")

        plt.tight_layout()
        plot_path = path['image_dir'] / "imagenette_training_metrics.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()

        return model


@task('data:evaluate_cnn_imagenette')
def evaluate_cnn_imagenette():
    path = get_project_paths()

    model_path = path['models_dir'] / "imagenette_cnn_model.pth"

    data = np.load(path['processed_dir'] / "imagenette.npz")

    X_test = torch.tensor(data["test_images"], dtype=torch.float32)
    y_test = torch.tensor(data["test_labels"], dtype=torch.long)

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Воссоздаем модель и загружаем веса
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())  # возвращаем на cpu для подсчета
            all_labels.extend(labels.cpu().numpy())  # то же самое

    # Логируем матрицу ошибок
    model_name_tag = 'imagenette'
    log_confusion_matrix(all_preds, all_labels, model_name_tag)

    # Вычисляем точность на тестовом наборе
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    return accuracy



@task('data:train_oxford_no_frease_nn')
def train_oxford_no_frease_nn():
    path = get_project_paths()

    # Загрузка данных
    data = np.load(path['processed_dir'] / "oxford.npz")
    X_train = torch.tensor(data["train_images"], dtype=torch.float32)
    y_train = torch.tensor(data["train_labels"], dtype=torch.long)
    X_val = torch.tensor(data["val_images"], dtype=torch.float32)
    y_val = torch.tensor(data["val_labels"], dtype=torch.long)

    # Даталоадеры
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

    # === Обучение ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    pretrained_model_path = path['models_dir'] / "imagenette_cnn_model.pth"
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))

    layers = list(model.net)
    in_features = layers[-1].in_features
    layers[-1] = nn.Linear(in_features, 37)
    model.net = nn.Sequential(*layers)

    model.to(device)

    # Оптимизатор: обучаем все параметры (без заморозки)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 250

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    patience = 50  # Количество эпох без улучшения, после которых остановим обучение
    best_val_loss = float('inf')
    epochs_since_improvement = 0

    with mlflow.start_run():
        for epoch in range(num_epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_train_loss = running_loss / total
            epoch_train_acc = correct / total
            train_losses.append(epoch_train_loss)
            train_accuracies.append(epoch_train_acc)

            # Валидация
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            epoch_val_loss = val_loss / val_total
            epoch_val_acc = val_correct / val_total
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_acc)

            # Проверка на улучшение
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if epochs_since_improvement >= patience:
                print("Early stopping triggered")
                break

            # Логирование метрик в MLflow
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_train_acc, step=epoch)
            mlflow.log_metric("val_accuracy", epoch_val_acc, step=epoch)

            print(f"[{epoch + 1}/{num_epochs}] Train Loss: {epoch_train_loss:.4f}, "
                  f"Val Loss: {epoch_val_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
                  f"Val Acc: {epoch_val_acc:.4f}")

        # === Сохраняем модель ===
        model_path = path['models_dir']/"oxford_no_frease_nn_model.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)

        # === Графики ===
        num_epochs_actual = epoch + 1 if epochs_since_improvement < patience else epoch

        # Подготовим графики с реальным количеством эпох
        epochs = range(1, num_epochs_actual + 1)

        # Графики
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses[:num_epochs_actual], label="Train Loss")
        plt.plot(epochs, val_losses[:num_epochs_actual], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss")

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies[:num_epochs_actual], label="Train Accuracy")
        plt.plot(epochs, val_accuracies[:num_epochs_actual], label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy")

        plt.tight_layout()
        plot_path = path['image_dir'] / "oxford_no_frease_training_metrics.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()

        return model



@task('data:evaluate_oxford_no_frease_model_on_test')
def evaluate_oxford_no_frease_model_on_test():
    path = get_project_paths()

    model_path = path['models_dir'] / "oxford_no_frease_nn_model.pth"

    data = np.load(path['processed_dir'] / "oxford.npz")

    X_test = torch.tensor(data["test_images"], dtype=torch.float32)
    y_test = torch.tensor(data["test_labels"], dtype=torch.long)

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Воссоздаем модель и загружаем веса
    model = SimpleCNN()
    layers = list(model.net)
    in_features = layers[-1].in_features
    layers[-1] = nn.Linear(in_features, 37)
    model.net = nn.Sequential(*layers)

    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())  # возвращаем на cpu для подсчета
            all_labels.extend(labels.cpu().numpy())  # то же самое

    # Логируем матрицу ошибок
    model_name_tag = 'mnist_nofrease'
    log_confusion_matrix(all_preds, all_labels, model_name_tag)

    # Вычисляем точность на тестовом наборе
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    return accuracy


@task('data:train_oxford_full_frease_nn')
def train_oxford_full_frease_nn():
    path = get_project_paths()

    # Загрузка данных
    data = np.load(path['processed_dir'] / "oxford.npz")
    X_train = torch.tensor(data["train_images"], dtype=torch.float32)
    y_train = torch.tensor(data["train_labels"], dtype=torch.long)
    X_val = torch.tensor(data["val_images"], dtype=torch.float32)
    y_val = torch.tensor(data["val_labels"], dtype=torch.long)

    # Даталоадеры
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

    # === Обучение ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    pretrained_model_path = path['models_dir'] / "imagenette_cnn_model.pth"
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))

    layers = list(model.net)
    in_features = layers[-1].in_features
    layers[-1] = nn.Linear(in_features, 37)
    model.net = nn.Sequential(*layers)

    # Замораживаем все слои
    for param in model.parameters():
        param.requires_grad = False

    # Размораживаем только последний слой
    for param in model.net[-1].parameters():
        param.requires_grad = True

    model.to(device)

    # Оптимизатор: обучаем все параметры (без заморозки)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 250

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    patience = 50  # Количество эпох без улучшения, после которых остановим обучение
    best_val_loss = float('inf')
    epochs_since_improvement = 0

    with mlflow.start_run():
        for epoch in range(num_epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_train_loss = running_loss / total
            epoch_train_acc = correct / total
            train_losses.append(epoch_train_loss)
            train_accuracies.append(epoch_train_acc)

            # Валидация
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            epoch_val_loss = val_loss / val_total
            epoch_val_acc = val_correct / val_total
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_acc)

            # Проверка на улучшение
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if epochs_since_improvement >= patience:
                print("Early stopping triggered")
                break

            # Логирование метрик в MLflow
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_train_acc, step=epoch)
            mlflow.log_metric("val_accuracy", epoch_val_acc, step=epoch)

            print(f"[{epoch + 1}/{num_epochs}] Train Loss: {epoch_train_loss:.4f}, "
                  f"Val Loss: {epoch_val_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
                  f"Val Acc: {epoch_val_acc:.4f}")

        # === Сохраняем модель ===
        model_path = path['models_dir']/"oxford_full_frease_nn_model.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)

        # === Графики ===
        num_epochs_actual = epoch + 1 if epochs_since_improvement < patience else epoch

        # Подготовим графики с реальным количеством эпох
        epochs = range(1, num_epochs_actual + 1)

        # Графики
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses[:num_epochs_actual], label="Train Loss")
        plt.plot(epochs, val_losses[:num_epochs_actual], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss")

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies[:num_epochs_actual], label="Train Accuracy")
        plt.plot(epochs, val_accuracies[:num_epochs_actual], label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy")

        plt.tight_layout()
        plot_path = path['image_dir'] / "oxford_full_frease_training_metrics.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()

        return model



@task('data:evaluate_oxford_full_frease_model_on_test')
def evaluate_oxford_full_frease_model_on_test():
    path = get_project_paths()

    model_path = path['models_dir'] / "oxford_full_frease_nn_model.pth"

    data = np.load(path['processed_dir'] / "oxford.npz")

    X_test = torch.tensor(data["test_images"], dtype=torch.float32)
    y_test = torch.tensor(data["test_labels"], dtype=torch.long)

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Воссоздаем модель и загружаем веса
    model = SimpleCNN()
    layers = list(model.net)
    in_features = layers[-1].in_features
    layers[-1] = nn.Linear(in_features, 37)
    model.net = nn.Sequential(*layers)

    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())  # возвращаем на cpu для подсчета
            all_labels.extend(labels.cpu().numpy())  # то же самое

    # Логируем матрицу ошибок
    model_name_tag = 'oxford_full_frease'
    log_confusion_matrix(all_preds, all_labels, model_name_tag)

    # Вычисляем точность на тестовом наборе
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    return accuracy



@task('data:train_oxford_part_frease_nn')
def train_oxford_part_frease_nn():
    path = get_project_paths()

    # Загрузка данных
    data = np.load(path['processed_dir'] / "oxford.npz")
    X_train = torch.tensor(data["train_images"], dtype=torch.float32)
    y_train = torch.tensor(data["train_labels"], dtype=torch.long)
    X_val = torch.tensor(data["val_images"], dtype=torch.float32)
    y_val = torch.tensor(data["val_labels"], dtype=torch.long)

    # Даталоадеры
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

    # === Обучение ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    pretrained_model_path = path['models_dir'] / "imagenette_cnn_model.pth"
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))

    layers = list(model.net)
    in_features = layers[-1].in_features
    layers[-1] = nn.Linear(in_features, 37)
    model.net = nn.Sequential(*layers)

    # Замораживаем все слои
    for param in model.parameters():
        param.requires_grad = False

    # Размораживаем только последний слой
    for param in model.net[-1].parameters():
        param.requires_grad = True

    model.to(device)

    # Оптимизатор: обучаем все параметры (без заморозки)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 250

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    patience = 50  # Количество эпох без улучшения, после которых остановим обучение
    best_val_loss = float('inf')
    epochs_since_improvement = 0
    unfreeze_at = patience // 2

    with mlflow.start_run():
        for epoch in range(num_epochs):
            model.train()
            if epoch == unfreeze_at:
                print(f"🔓 Размораживаем все слои на эпохе {epoch}")
                for param in model.parameters():
                    param.requires_grad = True
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Можно уменьшить lr

            running_loss, correct, total = 0.0, 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_train_loss = running_loss / total
            epoch_train_acc = correct / total
            train_losses.append(epoch_train_loss)
            train_accuracies.append(epoch_train_acc)

            # Валидация
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            epoch_val_loss = val_loss / val_total
            epoch_val_acc = val_correct / val_total
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_acc)

            # Проверка на улучшение
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if epochs_since_improvement >= patience:
                print("Early stopping triggered")
                break

            # Логирование метрик в MLflow
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_train_acc, step=epoch)
            mlflow.log_metric("val_accuracy", epoch_val_acc, step=epoch)

            print(f"[{epoch + 1}/{num_epochs}] Train Loss: {epoch_train_loss:.4f}, "
                  f"Val Loss: {epoch_val_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
                  f"Val Acc: {epoch_val_acc:.4f}")

        # === Сохраняем модель ===
        model_path = path['models_dir']/"oxford_part_frease_nn_model.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)

        # === Графики ===
        num_epochs_actual = epoch + 1 if epochs_since_improvement < patience else epoch

        # Подготовим графики с реальным количеством эпох
        epochs = range(1, num_epochs_actual + 1)

        # Графики
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses[:num_epochs_actual], label="Train Loss")
        plt.plot(epochs, val_losses[:num_epochs_actual], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss")

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies[:num_epochs_actual], label="Train Accuracy")
        plt.plot(epochs, val_accuracies[:num_epochs_actual], label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy")

        plt.tight_layout()
        plot_path = path['image_dir'] / "oxford_part_frease_training_metrics.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()

        return model



@task('data:evaluate_oxford_part_frease_model_on_test')
def evaluate_oxford_part_frease_model_on_test():
    path = get_project_paths()

    model_path = path['models_dir'] / "oxford_part_frease_nn_model.pth"

    data = np.load(path['processed_dir'] / "oxford.npz")

    X_test = torch.tensor(data["test_images"], dtype=torch.float32)
    y_test = torch.tensor(data["test_labels"], dtype=torch.long)

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Воссоздаем модель и загружаем веса
    model = SimpleCNN()
    layers = list(model.net)
    in_features = layers[-1].in_features
    layers[-1] = nn.Linear(in_features, 37)
    model.net = nn.Sequential(*layers)

    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())  # возвращаем на cpu для подсчета
            all_labels.extend(labels.cpu().numpy())  # то же самое

    # Логируем матрицу ошибок
    model_name_tag = 'oxford_part_frease'
    log_confusion_matrix(all_preds, all_labels, model_name_tag)

    # Вычисляем точность на тестовом наборе
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    return accuracy





@task('data:train_oxford_resnet18')
def train_oxford_resnet18():
    path = get_project_paths()

    # Загрузка данных
    data = np.load(path['processed_dir'] / "oxford.npz")
    X_train = torch.tensor(data["train_images"], dtype=torch.float32)
    y_train = torch.tensor(data["train_labels"], dtype=torch.long)
    X_val = torch.tensor(data["val_images"], dtype=torch.float32)
    y_val = torch.tensor(data["val_labels"], dtype=torch.long)

    # Преобразуем изображения в 3 канала для ResNet18 (ожидает 3xHxW)
    if X_train.shape[1] == 1:
        X_train = X_train.repeat(1, 3, 1, 1)
        X_val = X_val.repeat(1, 3, 1, 1)

    # Даталоадеры
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загружаем предобученную модель
    model = resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 37)  # Под задачи Oxford (37 классов)
    model = model.to(device)

    # Замораживаем все слои
    for param in model.parameters():
        param.requires_grad = False

    # Размораживаем последний слой
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 250
    patience = 50
    unfreeze_at = patience // 2
    best_val_loss = float('inf')
    epochs_since_improvement = 0

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    with mlflow.start_run():
        for epoch in range(num_epochs):
            model.train()

            if epoch == unfreeze_at:
                print(f"🔓 Размораживаем все слои на эпохе {epoch}")
                for param in model.parameters():
                    param.requires_grad = True
                optimizer = Adam(model.parameters(), lr=1e-4)

            running_loss, correct, total = 0.0, 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_train_loss = running_loss / total
            epoch_train_acc = correct / total
            train_losses.append(epoch_train_loss)
            train_accuracies.append(epoch_train_acc)

            # === Валидация ===
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            epoch_val_loss = val_loss / val_total
            epoch_val_acc = val_correct / val_total
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_acc)

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if epochs_since_improvement >= patience:
                print("Early stopping triggered")
                break

            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_train_acc, step=epoch)
            mlflow.log_metric("val_accuracy", epoch_val_acc, step=epoch)

            print(f"[{epoch + 1}/{num_epochs}] Train Loss: {epoch_train_loss:.4f}, "
                  f"Val Loss: {epoch_val_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
                  f"Val Acc: {epoch_val_acc:.4f}")

        # Сохранение модели
        model_path = path['models_dir'] / "resnet18_oxford.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)

        # Графики
        epochs = range(1, len(train_losses) + 1)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss")

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, label="Train Accuracy")
        plt.plot(epochs, val_accuracies, label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy")

        plt.tight_layout()
        plot_path = path['image_dir'] / "resnet18_oxford_metrics.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()

        return model



@task('data:evaluate_resnet18_oxford_part_frease_model_on_test')
def evaluate_resnet18_oxford_part_frease_model_on_test():
    path = get_project_paths()

    model_path = path['models_dir'] / "resnet18_oxford.pth"

    data = np.load(path['processed_dir'] / "oxford.npz")

    X_test = torch.tensor(data["test_images"], dtype=torch.float32)
    y_test = torch.tensor(data["test_labels"], dtype=torch.long)

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Воссоздаем модель и загружаем веса
    model = resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 37)  # Под задачи Oxford (37 классов)
    model = model.to(device)

    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())  # возвращаем на cpu для подсчета
            all_labels.extend(labels.cpu().numpy())  # то же самое

    # Логируем матрицу ошибок
    model_name_tag = 'resnet18_oxford'
    log_confusion_matrix(all_preds, all_labels, model_name_tag)

    # Вычисляем точность на тестовом наборе
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    return accuracy






@task('data:train_animal_no_freese')
def train_animal_no_freese():
    path = get_project_paths()

    # Загрузка данных
    data = np.load(path['processed_dir'] / "animals.npz")
    X_train = torch.tensor(data["train_images"], dtype=torch.float32)
    y_train = torch.tensor(data["train_labels"], dtype=torch.long)
    X_val = torch.tensor(data["val_images"], dtype=torch.float32)
    y_val = torch.tensor(data["val_labels"], dtype=torch.long)

    # Даталоадеры
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

    # === Обучение ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    pretrained_model_path = path['models_dir'] / "imagenette_cnn_model.pth"
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))

    layers = list(model.net)
    in_features = layers[-1].in_features
    layers[-1] = nn.Linear(in_features, 10)
    model.net = nn.Sequential(*layers)

    model.to(device)

    # Оптимизатор: обучаем все параметры (без заморозки)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 250

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    patience = 10  # Количество эпох без улучшения, после которых остановим обучение
    best_val_loss = float('inf')
    epochs_since_improvement = 0

    with mlflow.start_run():
        for epoch in range(num_epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_train_loss = running_loss / total
            epoch_train_acc = correct / total
            train_losses.append(epoch_train_loss)
            train_accuracies.append(epoch_train_acc)

            # Валидация
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            epoch_val_loss = val_loss / val_total
            epoch_val_acc = val_correct / val_total
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_acc)

            # Проверка на улучшение
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if epochs_since_improvement >= patience:
                print("Early stopping triggered")
                break

            # Логирование метрик в MLflow
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_train_acc, step=epoch)
            mlflow.log_metric("val_accuracy", epoch_val_acc, step=epoch)

            print(f"[{epoch + 1}/{num_epochs}] Train Loss: {epoch_train_loss:.4f}, "
                  f"Val Loss: {epoch_val_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
                  f"Val Acc: {epoch_val_acc:.4f}")

        # === Сохраняем модель ===
        model_path = path['models_dir']/"animals_no_frease_nn_model.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)

        # === Графики ===
        num_epochs_actual = epoch + 1 if epochs_since_improvement < patience else epoch

        # Подготовим графики с реальным количеством эпох
        epochs = range(1, num_epochs_actual + 1)

        # Графики
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses[:num_epochs_actual], label="Train Loss")
        plt.plot(epochs, val_losses[:num_epochs_actual], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss")

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies[:num_epochs_actual], label="Train Accuracy")
        plt.plot(epochs, val_accuracies[:num_epochs_actual], label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy")

        plt.tight_layout()
        plot_path = path['image_dir'] / "animals_no_frease_training_metrics.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()

        return model



@task('data:eveluate_animals_no_freease')
def eveluate_animals_no_freease():
    path = get_project_paths()

    model_path = path['models_dir'] / "animals_no_frease_nn_model.pth"

    data = np.load(path['processed_dir'] / "animals.npz")

    X_test = torch.tensor(data["test_images"], dtype=torch.float32)
    y_test = torch.tensor(data["test_labels"], dtype=torch.long)

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Воссоздаем модель и загружаем веса
    model = SimpleCNN()
    layers = list(model.net)
    in_features = layers[-1].in_features
    layers[-1] = nn.Linear(in_features, 10)
    model.net = nn.Sequential(*layers)

    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())  # возвращаем на cpu для подсчета
            all_labels.extend(labels.cpu().numpy())  # то же самое

    # Логируем матрицу ошибок
    model_name_tag = 'mnist_nofrease'
    log_confusion_matrix(all_preds, all_labels, model_name_tag)

    # Вычисляем точность на тестовом наборе
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    return accuracy



@task('data:train_animal_full_freese')
def train_animal_full_freese():
    path = get_project_paths()

    # Загрузка данных
    data = np.load(path['processed_dir'] / "animals.npz")
    X_train = torch.tensor(data["train_images"], dtype=torch.float32)
    y_train = torch.tensor(data["train_labels"], dtype=torch.long)
    X_val = torch.tensor(data["val_images"], dtype=torch.float32)
    y_val = torch.tensor(data["val_labels"], dtype=torch.long)

    # Даталоадеры
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

    # === Обучение ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    pretrained_model_path = path['models_dir'] / "imagenette_cnn_model.pth"
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))

    layers = list(model.net)
    in_features = layers[-1].in_features
    layers[-1] = nn.Linear(in_features, 10)
    model.net = nn.Sequential(*layers)

    # Замораживаем все слои
    for param in model.parameters():
        param.requires_grad = False

    # Размораживаем только последний слой
    for param in model.net[-1].parameters():
        param.requires_grad = True

    model.to(device)

    # Оптимизатор: обучаем все параметры (без заморозки)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 250

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    patience = 10  # Количество эпох без улучшения, после которых остановим обучение
    best_val_loss = float('inf')
    epochs_since_improvement = 0

    with mlflow.start_run():
        for epoch in range(num_epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_train_loss = running_loss / total
            epoch_train_acc = correct / total
            train_losses.append(epoch_train_loss)
            train_accuracies.append(epoch_train_acc)

            # Валидация
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            epoch_val_loss = val_loss / val_total
            epoch_val_acc = val_correct / val_total
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_acc)

            # Проверка на улучшение
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if epochs_since_improvement >= patience:
                print("Early stopping triggered")
                break

            # Логирование метрик в MLflow
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_train_acc, step=epoch)
            mlflow.log_metric("val_accuracy", epoch_val_acc, step=epoch)

            print(f"[{epoch + 1}/{num_epochs}] Train Loss: {epoch_train_loss:.4f}, "
                  f"Val Loss: {epoch_val_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
                  f"Val Acc: {epoch_val_acc:.4f}")

        # === Сохраняем модель ===
        model_path = path['models_dir']/"animals_full_frease_nn_model.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)

        # === Графики ===
        num_epochs_actual = epoch + 1 if epochs_since_improvement < patience else epoch

        # Подготовим графики с реальным количеством эпох
        epochs = range(1, num_epochs_actual + 1)

        # Графики
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses[:num_epochs_actual], label="Train Loss")
        plt.plot(epochs, val_losses[:num_epochs_actual], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss")

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies[:num_epochs_actual], label="Train Accuracy")
        plt.plot(epochs, val_accuracies[:num_epochs_actual], label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy")

        plt.tight_layout()
        plot_path = path['image_dir'] / "animals_full_frease_training_metrics.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()

        return model



@task('data:eveluate_animals_full_freease')
def eveluate_animals_full_freease():
    path = get_project_paths()

    model_path = path['models_dir'] / "animals_full_frease_nn_model.pth"

    data = np.load(path['processed_dir'] / "animals.npz")

    X_test = torch.tensor(data["test_images"], dtype=torch.float32)
    y_test = torch.tensor(data["test_labels"], dtype=torch.long)

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Воссоздаем модель и загружаем веса
    model = SimpleCNN()
    layers = list(model.net)
    in_features = layers[-1].in_features
    layers[-1] = nn.Linear(in_features, 10)
    model.net = nn.Sequential(*layers)

    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())  # возвращаем на cpu для подсчета
            all_labels.extend(labels.cpu().numpy())  # то же самое

    # Логируем матрицу ошибок
    model_name_tag = 'animals_full_freese'
    log_confusion_matrix(all_preds, all_labels, model_name_tag)

    # Вычисляем точность на тестовом наборе
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    return accuracy









@task('data:train_animal_part_freese')
def train_animal_part_freese():
    path = get_project_paths()

    # Загрузка данных
    data = np.load(path['processed_dir'] / "animals.npz")
    X_train = torch.tensor(data["train_images"], dtype=torch.float32)
    y_train = torch.tensor(data["train_labels"], dtype=torch.long)
    X_val = torch.tensor(data["val_images"], dtype=torch.float32)
    y_val = torch.tensor(data["val_labels"], dtype=torch.long)

    # Даталоадеры
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

    # === Обучение ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    pretrained_model_path = path['models_dir'] / "imagenette_cnn_model.pth"
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))

    layers = list(model.net)
    in_features = layers[-1].in_features
    layers[-1] = nn.Linear(in_features, 10)
    model.net = nn.Sequential(*layers)

    # Замораживаем все слои
    for param in model.parameters():
        param.requires_grad = False

    # Размораживаем только последний слой
    for param in model.net[-1].parameters():
        param.requires_grad = True

    model.to(device)

    # Оптимизатор: обучаем все параметры (без заморозки)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 250

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    patience = 10  # Количество эпох без улучшения, после которых остановим обучение
    best_val_loss = float('inf')
    epochs_since_improvement = 0
    unfreeze_at = 25

    with mlflow.start_run():
        for epoch in range(num_epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0

            if epoch == unfreeze_at:
                print(f"🔓 Размораживаем все слои на эпохе {epoch}")
                for param in model.parameters():
                    param.requires_grad = True
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_train_loss = running_loss / total
            epoch_train_acc = correct / total
            train_losses.append(epoch_train_loss)
            train_accuracies.append(epoch_train_acc)

            # Валидация
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            epoch_val_loss = val_loss / val_total
            epoch_val_acc = val_correct / val_total
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_acc)

            # Проверка на улучшение
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if epochs_since_improvement >= patience:
                print("Early stopping triggered")
                break

            # Логирование метрик в MLflow
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_train_acc, step=epoch)
            mlflow.log_metric("val_accuracy", epoch_val_acc, step=epoch)

            print(f"[{epoch + 1}/{num_epochs}] Train Loss: {epoch_train_loss:.4f}, "
                  f"Val Loss: {epoch_val_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
                  f"Val Acc: {epoch_val_acc:.4f}")

        # === Сохраняем модель ===
        model_path = path['models_dir']/"animals_part_frease_nn_model.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)

        # === Графики ===
        num_epochs_actual = epoch + 1 if epochs_since_improvement < patience else epoch

        # Подготовим графики с реальным количеством эпох
        epochs = range(1, num_epochs_actual + 1)

        # Графики
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses[:num_epochs_actual], label="Train Loss")
        plt.plot(epochs, val_losses[:num_epochs_actual], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss")

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies[:num_epochs_actual], label="Train Accuracy")
        plt.plot(epochs, val_accuracies[:num_epochs_actual], label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy")

        plt.tight_layout()
        plot_path = path['image_dir'] / "animals_part_frease_training_metrics.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()

        return model



@task('data:eveluate_animals_part_freease')
def eveluate_animals_part_freease():
    path = get_project_paths()

    model_path = path['models_dir'] / "animals_part_frease_nn_model.pth"

    data = np.load(path['processed_dir'] / "animals.npz")

    X_test = torch.tensor(data["test_images"], dtype=torch.float32)
    y_test = torch.tensor(data["test_labels"], dtype=torch.long)

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Воссоздаем модель и загружаем веса
    model = SimpleCNN()
    layers = list(model.net)
    in_features = layers[-1].in_features
    layers[-1] = nn.Linear(in_features, 10)
    model.net = nn.Sequential(*layers)

    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())  # возвращаем на cpu для подсчета
            all_labels.extend(labels.cpu().numpy())  # то же самое

    # Логируем матрицу ошибок
    model_name_tag = 'animals_part_freese'
    log_confusion_matrix(all_preds, all_labels, model_name_tag)

    # Вычисляем точность на тестовом наборе
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    return accuracy






@task('data:train_animals_resnet18')
def train_animals_resnet18():
    path = get_project_paths()

    # Загрузка данных
    data = np.load(path['processed_dir'] / "animals.npz")
    X_train = torch.tensor(data["train_images"], dtype=torch.float32)
    y_train = torch.tensor(data["train_labels"], dtype=torch.long)
    X_val = torch.tensor(data["val_images"], dtype=torch.float32)
    y_val = torch.tensor(data["val_labels"], dtype=torch.long)

    # Преобразуем изображения в 3 канала для ResNet18 (ожидает 3xHxW)
    if X_train.shape[1] == 1:
        X_train = X_train.repeat(1, 3, 1, 1)
        X_val = X_val.repeat(1, 3, 1, 1)

    # Даталоадеры
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загружаем предобученную модель
    model = resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    model = model.to(device)

    # Замораживаем все слои
    for param in model.parameters():
        param.requires_grad = False

    # Размораживаем последний слой
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 250
    patience = 20
    unfreeze_at = 25
    best_val_loss = float('inf')
    epochs_since_improvement = 0

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    with mlflow.start_run():
        for epoch in range(num_epochs):
            model.train()

            if epoch == unfreeze_at:
                print(f"🔓 Размораживаем все слои на эпохе {epoch}")
                for param in model.parameters():
                    param.requires_grad = True
                optimizer = Adam(model.parameters(), lr=1e-4)

            running_loss, correct, total = 0.0, 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_train_loss = running_loss / total
            epoch_train_acc = correct / total
            train_losses.append(epoch_train_loss)
            train_accuracies.append(epoch_train_acc)

            # === Валидация ===
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            epoch_val_loss = val_loss / val_total
            epoch_val_acc = val_correct / val_total
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_acc)

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if epochs_since_improvement >= patience:
                print("Early stopping triggered")
                break

            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_train_acc, step=epoch)
            mlflow.log_metric("val_accuracy", epoch_val_acc, step=epoch)

            print(f"[{epoch + 1}/{num_epochs}] Train Loss: {epoch_train_loss:.4f}, "
                  f"Val Loss: {epoch_val_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
                  f"Val Acc: {epoch_val_acc:.4f}")

        # Сохранение модели
        model_path = path['models_dir'] / "resnet18_animals.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)

        # Графики
        epochs = range(1, len(train_losses) + 1)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss")

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, label="Train Accuracy")
        plt.plot(epochs, val_accuracies, label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy")

        plt.tight_layout()
        plot_path = path['image_dir'] / "resnet18_animals_metrics.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()

        return model



@task('data:evaluate_resnet18_animals')
def evaluate_resnet18_animals():
    path = get_project_paths()

    model_path = path['models_dir'] / "resnet18_animals.pth"

    data = np.load(path['processed_dir'] / "animals.npz")

    X_test = torch.tensor(data["test_images"], dtype=torch.float32)
    y_test = torch.tensor(data["test_labels"], dtype=torch.long)

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Воссоздаем модель и загружаем веса
    model = resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    model = model.to(device)

    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())  # возвращаем на cpu для подсчета
            all_labels.extend(labels.cpu().numpy())  # то же самое

    # Логируем матрицу ошибок
    model_name_tag = 'resnet18_animals'
    log_confusion_matrix(all_preds, all_labels, model_name_tag)

    # Вычисляем точность на тестовом наборе
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    return accuracy


















if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", help="Список задач для выполнения")
    args = parser.parse_args()

    if args.tasks:
        main(args.tasks)  # Здесь передаем задачи, которые указаны в командной строке
