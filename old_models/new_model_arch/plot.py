import matplotlib.pyplot as plt

def plot_metrics(history):
    epochs = range(1, len(history['accuracy']) + 1)

    # Accuracy plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['accuracy'], 'b', label='Training accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_plot.png', dpi=300)
    plt.show()

    # AUC plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['auc'], 'r', label='Training AUC')
    plt.title('Training AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)
    plt.savefig('auc_plot.png', dpi=300)
    plt.show()

    # Loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['loss'], 'g', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png', dpi=300)
    plt.show()
