import logging
import pathlib
import statistics
import sys


import matplotlib.pyplot as pyplot
import torch


LOGGER = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
LOGGER.addHandler(stream_handler)
LOGGER.setLevel(logging.DEBUG)


def main():
    window = 20
    max_loss = 50
    lrs = [str(1.0 / (2**i)) for i in range(13)]
    linestyles = [color + line
                  for line in ['-', '--', '-.', ':']
                  for color in ['b', 'g', 'r', 'c', 'm', 'y', 'k']]
    legend = []
    for lr, linestyle in zip(lrs, linestyles):
        metrics_path = pathlib.Path('models/densenet121/hypersearch_subset16000_lr' + lr + '/metrics')
        if metrics_path.exists() and metrics_path.is_file():
            d = torch.load(str(metrics_path))
            training_loss = d['training_loss']
            training_loss = [min(x, max_loss) for x in training_loss]
            smoothed_loss = [statistics.median(training_loss[i:i + window])
                             for i in range(0, len(training_loss) - window, window)]
            pyplot.plot(range(0, len(training_loss) - window, window), smoothed_loss, linestyle)
            legend.append('lr = {}'.format(lr))
        else:
            LOGGER.debug('Skipping {}.'.format(metrics_path))

    pyplot.legend(legend)
    pyplot.show()


if __name__ == '__main__':
    main()
