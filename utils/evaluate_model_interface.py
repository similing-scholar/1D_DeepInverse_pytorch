import matplotlib.pyplot as plt


def plot_metric(df_history, metric):
    train_metrics = df_history["train_"+metric]
    val_metrics = df_history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)

    fig = plt.figure()
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])

    plt.tight_layout()
    plt.rcParams['font.sans-serif'] = ['simhei']
    plt.rcParams['axes.unicode_minus'] = False

    return fig