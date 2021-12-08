import argparse
from operator import itemgetter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('history_path', type=str, help="Checkpoint to load the weights from. ex: checkpoints/vae/vae.data-00000-of-00001 then "
                                                       "input checkpoints/vae/vae")
    parser.add_argument('-dfk', '--drop_first_k_epochs', type=int, default=2,
                        help='Number of epochs of metrics to drop from the start (since first few losses are extremely high, it can skew the graph) Ex: 2 (default).')
    args = vars(parser.parse_args())

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path


def plot_history_metric(metrics, history, savefiles=None):
    for metric in metrics:
        val = f'val_{metric}'
        plt.clf()
        plt.figure()
        plt.plot(history[metric])
        if val in history.keys():
            plt.plot(history[val])
        plt.title(f'{metric.capitalize()}')
        plt.ylabel(f'{metric.capitalize()}')
        plt.xlabel('Epoch')
        if val in history.keys():
            plt.legend(['train', 'val'], loc='upper left')
        if not savefiles:
            plt.show()


def main(ar):
    history_path, drop_first_k_epochs = itemgetter('history_path', 'drop_first_k_epochs')(ar)
    if Path(history_path).is_file():
        with open(history_path, 'r') as f:
            history_json = json.load(f)
            for metric in history_json:
                metric_data = np.array(history_json[metric])
                metric_data = metric_data[drop_first_k_epochs:]
                history_json[metric] = metric_data

            plot_history_metric(['loss', 'reconstruction_loss', 'kl_loss', 'perception_loss'], history_json)


if __name__ == '__main__':
    main(args)
