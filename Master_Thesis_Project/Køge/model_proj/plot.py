import matplotlib
from matplotlib import pyplot as plt
from datetime import datetime, date
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
#plt.style.use('ggplot')

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

def show_batch(image, label):
    #plt.figure(figsize=(10,10))


    batch_image = unpack_tensor(image)
    labels = define_label_list(label)
    fig, big_axes = plt.subplots( figsize=(10.0, 10.0) , nrows=3, ncols=1, sharey=True)
    np.vectorize(lambda ax:ax.axis('off'))(big_axes) 
    
    
    plt.suptitle("Custom Generator Batch View", fontweight="bold", fontsize='large')

    big_axes[2].set_title("EEG ALL Frequencies", pad=30, fontweight="bold", fontsize='medium')
    big_axes[0].set_title("EEG 10 > Frequencies", pad=30, fontweight="bold", fontsize='medium')
    big_axes[1].set_title("ECG", pad=30, fontweight="bold", fontsize='medium')

    for i in range(1,10):
        ax = fig.add_subplot(3,3,i)
        ax.axis('off')
        ax.title.set_text(str(labels[i-1]))
        plt.imshow(batch_image[i-1])
    

    fig.set_facecolor('w')
    plt.tight_layout()
    plt.axis('off')
    plt.show()
    



def define_label_list(label_batch):
    cont = []
    for i in range(3):
        for l in label_batch:
            cont.append(l)
    return cont

def unpack_tensor(img_tensor):
    container = np.array([x for x in img_tensor])
    container = container.reshape(9, 224, 224, 3)
    return container