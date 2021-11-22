import matplotlib
from matplotlib import pyplot as plt
from datetime import datetime, date
from sklearn.metrics import ConfusionMatrixDisplay

def evaluate_training_plot(history, path, same_plot=False):
    if same_plot:
        plt.figure(figsize=(3,4))
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,10))
        plt.tight_layout()
        ax1.set_title('acc')
        ax1.plot(history.history['categorical_accuracy'], color='red')
        ax1.plot(history.history['val_categorical_accuracy'], color='blue')
        ax2.set_title('loss')
        ax2.plot(history.history['loss'], color='red')
        ax2.plot(history.history['val_loss'], color='blue')
        plt.savefig(f"{path}{date.today()}_history_compare.png")

    plt.figure(figsize=(3,4))
    fig, axs = plt.subplots(2,2, figsize=(10,10))
    plt.tight_layout()
    axs[0,0].set_title('acc')
    axs[0,0].plot(history.history['categorical_accuracy'], color='red')
    axs[0,1].set_title('loss')
    axs[0,1].plot(history.history['loss'], color='red')
    axs[1,0].set_title('val_acc')
    axs[1,0].plot(history.history['val_categorical_accuracy'], color='red')
    axs[1,1].set_title('val_loss')
    axs[1,1].plot(history.history['val_loss'], color='red')
    plt.savefig(f"{path}{date.today()}_history_metrics_eval.png")

def plot_con_matrix(matrix, path, labels=None):
    ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels).plot(cmap='Blues')
    plt.title("Frozen Layers: 5 Epoch : four patient (all data)=train : validation split = test")
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(f"{path}{date.today()}_confus_matrix.png")