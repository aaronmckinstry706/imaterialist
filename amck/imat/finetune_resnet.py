import argparse
import collections
import datetime
import functools
import logging
import multiprocessing
import pathlib
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


# Set up the logging.

LOGGER = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
LOGGER.addHandler(stream_handler)

# For coordinating the display of plot data between threads.
metrics = collections.defaultdict(list)


def metrics_plotter():
    def plot_metrics(clargs):
        f = pyplot.figure()
        f.suptitle('Training Metrics')

        for p, (name, vals) in enumerate(metrics.items()):
            if len(vals) > 1:
                ax = pyplot.subplot(len(metrics), 2, 2 * p + 1)
                ax.plot(vals)
                ax.set_title(name)

                ax = pyplot.subplot(len(metrics), 2, 2 * p + 2)
                ax.plot(vals[-clargs.short_history:])
                ax.set_title(name)

        if len(metrics) > 0:
            pyplot.show(block=True)

    return plot_metrics


def command_listener(clargs):
    while True:
        command = input()
        if command == 'metrics':
            multiprocessing.Process(target=metrics_plotter(), args=(clargs,)).start()
        elif command == 'exit':
            exit(0)


def train(clargs):
    save_path = pathlib.Path(clargs.save_path)
    save_path.resolve()
    save_path = save_path.absolute()
    save_path.mkdir(parents=True, exist_ok=True)
    model_path = save_path / 'model'
    hyperparameters_path = save_path / 'hyperparameters'
    metrics_path = save_path / 'metrics'

    hyperparameters = dict()
    hyperparameters['clargs'] = vars(clargs)

    if clargs.pretrained:
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
    if clargs.training_subset < len(training_data):
        subset_indices = random.sample([i for i in range(len(training_data))], clargs.training_subset)
        training_data_sampler = sampler.SubsetRandomSampler(subset_indices)
    else:
        training_data_sampler = sampler.RandomSampler(training_data)
    training_data_loader = data.DataLoader(training_data, batch_size=clargs.batch_size,
                                           num_workers=clargs.num_workers, sampler=training_data_sampler)

    validation_data = datasets.ImageFolder('data/validation', transform=image_transform)
    validation_data_loader = data.DataLoader(validation_data, batch_size=clargs.batch_size,
                                             num_workers=clargs.num_workers)

    resnet: models.ResNet = models.resnet18(pretrained=clargs.pretrained)
    resnet.fc = nn.Linear(512, 128)
    resnet.cuda()

    optimizer = optim.SGD(resnet.fc.parameters(), lr=clargs.learning_rate, weight_decay=0.0001)
    hyperparameters['optimizer'] = optimizer.state_dict()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=1, verbose=True)

    training_loss_function = functional.cross_entropy
    validation_loss_function = functools.partial(functional.cross_entropy, size_average=False)

    torch.save(hyperparameters, str(hyperparameters_path))

    validation_stopwatch = stopwatch.Stopwatch()
    training_stopwatch = stopwatch.Stopwatch()
    saving_stopwatch = stopwatch.Stopwatch()
    best_validation_accuracy = -1.0
    with stopwatch.Stopwatch() as total_time_stopwatch, stopwatch.Stopwatch() as epoch_stopwatch:
        for i in range(clargs.epoch_limit):
            if scheduler.in_cooldown:
                metrics['learning_rate_drop_epoch'].append(i)

            LOGGER.debug('Training...')

            with training_stopwatch:
                epoch_loss_history, epoch_acccuracy_history = training.train(
                    training_data_loader, resnet, optimizer, training_loss_function, cuda=True)
                training_stopwatch.lap()

            LOGGER.debug('Validating...')

            with validation_stopwatch:
                validation_loss, validation_accuracy = training.evaluate_loss_and_accuracy(
                    validation_data_loader, resnet, validation_loss_function, cuda=True)
                validation_stopwatch.lap()

            metrics['training_loss'].extend(epoch_loss_history)
            metrics['validation_loss'].append(validation_loss)
            metrics['training_accuracy'].extend(epoch_acccuracy_history)
            metrics['validation_accuracy'].append(validation_accuracy)

            LOGGER.debug('Saving...')

            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                torch.save(resnet, str(model_path))
                saving_stopwatch.lap()

            torch.save(metrics, metrics_path)

            epoch_stopwatch.lap()

            LOGGER.info('epoch {epoch}\n'
                        '- total duration:                  {total_duration}\n'
                        '- validation loss:                 {validation_loss}\n'
                        '- validation accuracy:             {validation_accuracy}\n'
                        '- average training-batch loss:     {avg_training_loss}\n'
                        '- average training-batch accuracy: {avg_training_accuracy}'
                        .format(epoch=i,
                                total_duration=epoch_stopwatch.lap_times()[-1],
                                validation_loss=validation_loss,
                                validation_accuracy=validation_accuracy,
                                avg_training_loss=sum(epoch_loss_history)/float(len(epoch_loss_history)),
                                avg_training_accuracy=sum(epoch_acccuracy_history)/len(epoch_acccuracy_history)))
            LOGGER.debug('- training duration:              {training_duration}\n'
                         '- validation duration:            {validation_duration}'
                         .format(training_duration=training_stopwatch.lap_times()[-1],
                                 validation_duration=validation_stopwatch.lap_times()[-1]))

            scheduler.step(validation_accuracy)

        total_time_stopwatch.lap()

    print('total_time={}'.format(total_time_stopwatch.lap_times()[-1]))


def evaluate(clargs):
    if clargs.normalization == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # From Imagenet.
    elif clargs.normalization == 'imaterialist':
        # From iMaterialist.
        normalize = transforms.Normalize(mean=[0.6837, 0.6461, 0.6158], std=[0.2970, 0.3102, 0.3271])
    image_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    validation_data = datasets.ImageFolder('data/validation', transform=image_transform)
    validation_data_loader = data.DataLoader(validation_data, batch_size=clargs.batch_size,
                                             num_workers=clargs.num_workers)

    validation_loss_function = functools.partial(functional.cross_entropy, size_average=False)

    resnet = torch.load(clargs.resnet_model)
    resnet.cuda()

    validation_loss, validation_accuracy = training.evaluate_loss_and_accuracy(
        validation_data_loader, resnet, validation_loss_function, cuda=True)

    LOGGER.info('validation loss:       {}\n'
                'validation accuracy:   {}'
                .format(validation_loss, validation_accuracy))


def get_args():
    arg_parser = argparse.ArgumentParser(
        description='Trains Resnet-18 from the PyTorch models on the iMaterialist training data, and evaluates trained '
                    'resnet models on iMaterialist validation data.')
    arg_parser.add_argument(
        '--verbose', '-v', action='count', default=0,
        help='Indicate the level of verbosity. Include once to get info logs and twice to include debug-level logs.')
    arg_parser.add_argument('--num-workers', '-w', type=int, default=8,
                            help='Number of processes concurrently loading images from the training dataset.')
    arg_parser.add_argument('--batch-size', '-b', type=int, default=128, help='Batch size.')

    subparsers = arg_parser.add_subparsers(dest='subparser_name')

    train_subcommand_parser = subparsers.add_parser(
        'train', help='Trains a Resnet-18 model on the iMaterialist training data.')
    train_subcommand_parser.add_argument('--epoch-limit', '-l', type=int, default=41,
                                         help='The maximum number of epochs for which the network will train.')
    train_subcommand_parser.add_argument('--short-history', '-s', type=int, default=1500,
                                         help='The length of the short history to  display in the plot of metrics.')
    train_subcommand_parser.add_argument('--pretrained', '-p', action='store_true', default=False,
                                         help='Indicates that the model should be pretrained on ImageNet.')
    train_subcommand_parser.add_argument('--training-subset', '-t', type=int, default=sys.maxsize,
                                         help='Indicates that only TRAINING_SUBSET number of samples should be used for'
                                              ' training data.')
    train_subcommand_parser.add_argument('--learning-rate', '-r', type=float, default=0.001,
                                         help='Initial learning rate.')
    train_subcommand_parser.add_argument('--save-path', '-n', default=str(datetime.datetime.now()).replace(' ', '_'),
                                         help='The directory path under which to save all model results during '
                                              'training.')
    train_subcommand_parser.set_defaults(train=True)

    evaluate_subcommand_parser = subparsers.add_parser(
        'evaluate', help='Evaluates a given Resnet-18 model on the iMaterialist validation data.')
    evaluate_subcommand_parser.add_argument('--normalization', '-n', default='imaterialist',
                                            help='Normalization to use for evaluation.')
    evaluate_subcommand_parser.add_argument('model_directory', help='The model directory for the Resnet-18 model to be '
                                                                 'evaluated.')
    evaluate_subcommand_parser.set_defaults(evaluate=True)

    parsed_args = arg_parser.parse_args()

    # Check additional restrictions on values and types.

    if parsed_args.num_workers < 1:
        arg_parser.error("--num-workers must be positive.")
    if parsed_args.batch_size < 1:
        arg_parser.error("--batch-size must be positive.")

    if parsed_args.train:
        if parsed_args.epoch_limit < 1:
            arg_parser.error("--epoch-limit must be positive.")
        if parsed_args.short_history < 2:
            arg_parser.error("--short-history must be at least 2.")
        if parsed_args.training_subset < 1:
            arg_parser.error("--training-subset must be positive.")
        if parsed_args.learning_rate <= 0.0:
            arg_parser.error("--learning-rate must be positive.")
        save_path = pathlib.Path(parsed_args.save_path)
        if save_path.exists():
            arg_parser.error("--save-path must not already exist.")

    elif parsed_args.evaluate:
        load_path = pathlib.Path(parsed_args.model_directory)
        if not load_path.exists():
            arg_parser.error("model_directory must exist.")
        elif not load_path.is_dir():
            arg_parser.error("model_directory must be a directory.")

    return parsed_args


def main():

    parsed_args = get_args()

    if parsed_args.verbose == 1:
        LOGGER.setLevel(logging.INFO)
    elif parsed_args.verbose >= 2:
        LOGGER.setLevel(logging.DEBUG)

    if parsed_args.subparser_name == 'train':
        thread_target = train
    elif parsed_args.subparser_name == 'evaluate':
        thread_target = evaluate
    main_thread = threading.Thread(target=thread_target, args=(parsed_args,))
    main_thread.daemon = True
    main_thread.start()
    if parsed_args.subparser_name == 'train':
        command_thread = threading.Thread(target=command_listener, args=(parsed_args,))
        command_thread.start()
        command_thread.join()
    else:
        main_thread.join()


if __name__ == '__main__':
    main()
