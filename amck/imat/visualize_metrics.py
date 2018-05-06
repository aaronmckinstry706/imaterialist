import argparse
import pathlib
import sys


import matplotlib.pyplot as pyplot
import torch


def main(parsed_args):
    metrics = torch.load(parsed_args.training_directory + '/metrics')

    pyplot.subplot(4, 1, 1)
    pyplot.plot(metrics['training_loss'])
    pyplot.legend(['training loss'])

    pyplot.subplot(4, 1, 2)
    pyplot.plot(metrics['validation_loss'])
    pyplot.legend(['validation loss'])

    pyplot.subplot(4, 1, 3)
    pyplot.plot(metrics['training_accuracy'])
    pyplot.legend(['training accuracy'])

    pyplot.subplot(4, 1, 4)
    pyplot.plot(metrics['validation_accuracy'])
    pyplot.legend(['validation accuracy'])

    pyplot.show(block=True)


def get_args(raw_args):
    arg_parser = argparse.ArgumentParser(description='A script for visualizing the metrics output by the training '
                                                     'script.')
    arg_parser.add_argument('training_directory', help='The save directory path for the model output by the training '
                                                       'script.')
    parsed_args = arg_parser.parse_args(raw_args)

    training_directory = pathlib.Path(parsed_args.training_directory)
    if not training_directory.exists():
        arg_parser.error('training_directory must exist.')
    if not training_directory.is_dir():
        arg_parser.error('training_directory must be a directory.')

    return parsed_args


if __name__ == '__main__':
    main(get_args(sys.argv[1:]))
