from typing import Dict, List
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, filename: str):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "../train_images")
    plt.figure(figsize=(8, 4))
    plt.title(f"ROC-кривая", fontsize=12, pad=10)
    plt.xlabel(f"1 - специфичность", fontsize=10)
    plt.ylabel(f"чувствительность", fontsize=10, labelpad=10)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.plot(fpr, tpr, color="tomato")
    plt.plot([0, 1], [0, 1], color="skyblue", linestyle="--")
    plt.plot([0, 1], [1, 0], color="skyblue", linestyle="--")
    plt.savefig(
        os.path.join(output_dir, f"{filename}.png"), dpi=300, bbox_inches="tight"
    )


def plot_loss(history_data: Dict[str, List[float]], filename: str):
    train_loss = history_data["loss"]
    val_loss = history_data["val_loss"]
    train_accuracy = history_data["binary_accuracy"]
    val_accuracy = history_data["val_binary_accuracy"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    x_values = [i for i in range(1, len(train_loss) + 1)]
    # Первый график:
    axes[0, 0].plot(x_values, train_loss, label="Ошибка", color="skyblue")
    axes[0, 0].set_title("Функция потерь при обучении модели", pad=10)
    axes[0, 0].set_xlabel("Эпохи", labelpad=0)
    axes[0, 0].set_ylabel("Функция потерь", labelpad=10)
    axes[0, 0].legend()
    # Второй график:
    axes[1, 0].plot(x_values, train_accuracy, label="Точность", color="tomato")
    axes[1, 0].set_title("Точность модели во время обучения", pad=10)
    axes[1, 0].set_xlabel("Эпохи", labelpad=0)
    axes[1, 0].set_ylabel("Значение точности", labelpad=10)
    axes[1, 0].legend()
    # Третий график:
    axes[0, 1].plot(x_values, val_loss, label="Ошибка", color="skyblue")
    axes[0, 1].set_title("Функция потерь на проверочной выборке", pad=10)
    axes[0, 1].set_xlabel("Эпохи", labelpad=0)
    axes[0, 1].set_ylabel("Функция потерь", labelpad=10)
    axes[0, 1].legend()
    # Четвертый график:
    axes[1, 1].plot(x_values, val_accuracy, label="Точность", color="tomato")
    axes[1, 1].set_title("Точность модели на проверочной выборке", pad=10)
    axes[1, 1].set_xlabel("Эпохи", labelpad=1)
    axes[1, 1].set_ylabel("Значение точности", labelpad=10)
    axes[1, 1].legend()

    # Настройка отступов между графиками:
    fig.subplots_adjust(hspace=0.5)  # Отступ между графиками

    # Сохраняем график:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "../train_images")
    # plt.savefig("train_images/history_100-epochs_ferritin-10000-2.png", dpi=300)
    plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)


def plot_loss2(
    history_data: Dict[str, List[float]], softmax: bool = False, filename: str = ""
):
    metric = "categorical" if softmax else "binary"
    train_loss = history_data["loss"]
    val_loss = history_data["val_loss"]
    train_accuracy = history_data[f"{metric}_accuracy"]
    val_accuracy = history_data[f"val_{metric}_accuracy"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    x_values = [i for i in range(1, len(train_loss) + 1)]
    # Первый график:
    axes[0].plot(x_values, train_loss, label="Обучающая выборка", color="skyblue")
    axes[0].plot(x_values, val_loss, label="Валидационная выборка", color="orange")
    axes[0].set_title("Функция потерь", pad=10)
    axes[0].set_xlabel("Эпохи", labelpad=0)
    axes[0].set_ylabel("Величина ошибки", labelpad=10)
    axes[0].legend(fontsize="small", framealpha=0.5)
    # Второй график:
    axes[1].plot(x_values, train_accuracy, label="Обучающая выборка", color="skyblue")
    axes[1].plot(x_values, val_accuracy, label="Валидационная выборка", color="orange")
    axes[1].set_title("Точность модели", pad=10)
    axes[1].set_xlabel("Эпохи", labelpad=0)
    axes[1].set_ylabel("Значение точности", labelpad=10)
    axes[1].legend(fontsize="small", framealpha=0.5)

    # Настройка отступов между графиками:
    fig.subplots_adjust(hspace=0.5)  # Отступ между графиками

    # Сохраняем график:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "../train_images")
    plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
