import argparse
import collections
import datetime
import functools
import logging
import multiprocessing
import random
import sys
import threading
import typing


import matplotlib.pyplot as pyplot
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torch.utils.data.sampler as sampler
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms


import amck.imat.stopwatch as stopwatch
import amck.imat.training as training


# Parse command line arguments.

arg_parser = argparse.ArgumentParser(
    description='Trains Resnet-18 from the PyTorch models on the iMaterialist training data.')
arg_parser.add_argument('--num-workers', '-w', type=int, default=8,
                        help='Number of processes concurrently loading images from the training dataset.')
arg_parser.add_argument('--epoch-limit', '-l', type=int, default=41,
                        help='The maximum number of epochs for which the network will train.')
arg_parser.add_argument('--short-history', '-s', type=int, default=1500,
                        help='The length of the short history to  display in the plot of metrics.')
arg_parser.add_argument('--batch-size', '-b', type=int, default=256, help='Batch size.')
arg_parser.add_argument('--save-interval', '-i', type=int, default=500,
                        help='Every SAVE_INTERVAL iterations, a copy of the model will be saved.')
arg_parser.add_argument('--pretrained', '-p', action='store_true', default=False,
                        help='Indicates that the model should be pretrained on ImageNet.')
arg_parser.add_argument('--training-subset', '-t', type=int, default=sys.maxsize,
                        help='Indicates that only TRAINING_SUBSET number of samples should be used for training data.')
arg_parser.add_argument('--learning-rate', '-r', type=float, default=0.001, help='Initial learning rate.')
arg_parser.add_argument('--verbose', '-v', action='count', default=0,
                        help='Indicate the level of verbosity. Include once to get info logs and twice to include '
                             'debug-level logs.')
parsed_args = arg_parser.parse_args()

# Set up the logging.

LOGGER = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
LOGGER.addHandler(stream_handler)
if parsed_args.verbose == 1:
    LOGGER.setLevel(logging.INFO)
elif parsed_args.verbose >= 2:
    LOGGER.setLevel(logging.DEBUG)

metrics = collections.defaultdict(list)


def metrics_plotter():
    def plot_metrics():
        pyplot.figure()

        for p, (name, vals) in enumerate(metrics.items()):
            if len(vals) > 1:
                pyplot.subplot(len(metrics), 2, 2 * p + 1)
                pyplot.plot(vals)
                pyplot.subplot(len(metrics), 2, 2 * p + 2)
                pyplot.plot(vals[-parsed_args.short_history:])

        if len(metrics) > 0:
            pyplot.show(block=True)

    return plot_metrics


def command_listener():
    while True:
        command = input()
        if command == 'metrics':
            multiprocessing.Process(target=metrics_plotter()).start()
        elif command == 'exit':
            exit(0)


def main():
    START_TIME_STR = str(datetime.datetime.now()).replace(' ', '_')

    if parsed_args.pretrained:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # From Imagenet.
    else:
        # From iMaterialist.
        normalize = transforms.Normalize(mean=[0.6837, 0.6461, 0.6158], std=[0.2970, 0.3102, 0.3271])
    image_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    training_data = datasets.ImageFolder('data/training', transform=image_transform)
    if parsed_args.training_subset < len(training_data):
        subset_indices = random.sample([i for i in range(len(training_data))], parsed_args.training_subset)
        training_data_sampler = sampler.SubsetRandomSampler(subset_indices)
    else:
        training_data_sampler = sampler.RandomSampler(training_data)
    training_data_loader = data.DataLoader(training_data, batch_size=parsed_args.batch_size,
                                           num_workers=parsed_args.num_workers, sampler=training_data_sampler)

    validation_data = datasets.ImageFolder('data/validation', transform=image_transform)
    validation_data_loader = data.DataLoader(validation_data, batch_size=parsed_args.batch_size,
                                             num_workers=parsed_args.num_workers)

    resnet: models.ResNet = models.resnet18(pretrained=parsed_args.pretrained)
    resnet.fc = nn.Linear(512, 128)
    resnet.cuda()

    optimizer = optim.SGD(resnet.fc.parameters(), lr=parsed_args.learning_rate, weight_decay=0.0001, momentum=0.9)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=1, verbose=True)

    training_loss_function = functional.cross_entropy
    validation_loss_function = functools.partial(functional.cross_entropy, size_average=False)

    validation_stopwatch = stopwatch.Stopwatch()
    training_stopwatch = stopwatch.Stopwatch()
    saving_stopwatch = stopwatch.Stopwatch()
    best_validation_loss = float("inf")
    with stopwatch.Stopwatch() as total_time_stopwatch, stopwatch.Stopwatch() as epoch_stopwatch:
        for i in range(parsed_args.epoch_limit):
            LOGGER.debug('Training...')

            with training_stopwatch:
                epoch_loss_history = training.train(training_data_loader, resnet, optimizer, training_loss_function,
                                                    cuda=True)
                training_stopwatch.lap()

            LOGGER.debug('Validating...')

            with validation_stopwatch:
                validation_loss = training.validate(validation_data_loader, resnet, validation_loss_function, cuda=True)
                validation_stopwatch.lap()

            metrics['training_loss'].extend(epoch_loss_history)
            metrics['validation_loss'].append(validation_loss)

            LOGGER.debug('Saving...')

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save(resnet, 'models/finetuned_resnet_{}'.format(START_TIME_STR))
                saving_stopwatch.lap()

            epoch_stopwatch.lap()

            LOGGER.info('epoch {epoch}\n'
                        '- total duration:              {total_duration}\n'
                        '- validation loss:             {validation_loss}\n'
                        '- average training-batch loss: {avg_training_loss}'
                        .format(epoch=i,
                                total_duration=epoch_stopwatch.lap_times()[-1],
                                validation_loss=validation_loss,
                                avg_training_loss=sum(epoch_loss_history)/float(len(epoch_loss_history))))
            LOGGER.debug('- training duration:           {training_duration}\n'
                         '- validation duration:         {validation_duration}'
                         .format(training_duration=training_stopwatch.lap_times()[-1],
                                 validation_duration=validation_stopwatch.lap_times()[-1]))

            scheduler.step(validation_loss)

        total_time_stopwatch.lap()

    print('total_time={}'.format(total_time_stopwatch.lap_times()[-1]))
    torch.save(resnet, 'models/finetuned_resnet_' + START_TIME_STR)


if __name__ == '__main__':
    main_thread = threading.Thread(target=main)
    main_thread.daemon = True
    main_thread.start()
    command_thread = threading.Thread(target=command_listener)
    command_thread.start()
    command_thread.join()
