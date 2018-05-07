import argparse
import collections
import datetime
import functools
import logging
import pathlib
import random
import sys
import typing


import PIL.Image as Image
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


def train(clargs):
    metrics = collections.defaultdict(list)
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
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=15, scale=(1.0, 1.3), resample=Image.BILINEAR, fillcolor=2 ** 24 - 1),
        transforms.ColorJitter(brightness=0.2, contrast=0.8, saturation=0.8, hue=0.3),
        transforms.RandomGrayscale(p=0.1),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])

    training_data = datasets.ImageFolder('data/training', transform=image_transform)
    if clargs.training_subset < len(training_data):
        subset_indices = random.sample([i for i in range(len(training_data))], clargs.training_subset)
        training_data_sampler = sampler.SubsetRandomSampler(subset_indices)
    else:
        training_data_sampler = sampler.RandomSampler(training_data)
    training_data_loader = data.DataLoader(training_data, batch_size=clargs.training_batch_size,
                                           num_workers=clargs.num_workers, sampler=training_data_sampler)

    validation_data = datasets.ImageFolder('data/validation', transform=image_transform)
    validation_data_loader = data.DataLoader(validation_data, batch_size=clargs.validation_batch_size,
                                             num_workers=clargs.num_workers)

    network: models.DenseNet = models.densenet161(pretrained=clargs.pretrained)
    network.classifier = nn.Linear(network.classifier.in_features, 128)
    network.cuda()

    optimizer = optim.SGD(network.parameters(), lr=clargs.learning_rate, weight_decay=0.0001)
    hyperparameters['optimizer'] = optimizer.state_dict()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=0, verbose=True, eps=0)

    training_loss_function = functools.partial(functional.cross_entropy, size_average=False)
    validation_loss_function = functools.partial(functional.cross_entropy, size_average=False)

    torch.save(hyperparameters, str(hyperparameters_path))

    validation_stopwatch = stopwatch.Stopwatch()
    training_stopwatch = stopwatch.Stopwatch()
    saving_stopwatch = stopwatch.Stopwatch()
    best_validation_accuracy = -1.0
    with stopwatch.Stopwatch() as total_time_stopwatch, stopwatch.Stopwatch() as epoch_stopwatch:
        for i in range(clargs.epoch_limit):
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr < 10**-8:
                break
            metrics['lr'].append(current_lr)

            LOGGER.debug('Training...')

            with training_stopwatch:
                epoch_loss_history, epoch_acccuracy_history = training.train(
                    training_data_loader, network, optimizer, training_loss_function, cuda=True,
                    progress_bar=(clargs.verbose >= 2))
                training_stopwatch.lap()

            LOGGER.debug('Validating...')

            with validation_stopwatch:
                validation_loss, validation_accuracy = training.evaluate_loss_and_accuracy(
                    validation_data_loader, network, validation_loss_function, cuda=True,
                    progress_bar=(clargs.verbose >= 2))
                validation_stopwatch.lap()

            metrics['training_loss'].extend(
                [batch_loss / clargs.training_batch_size for batch_loss in epoch_loss_history])
            metrics['validation_loss'].append(validation_loss / len(validation_data))
            metrics['training_accuracy'].extend(epoch_acccuracy_history)
            metrics['validation_accuracy'].append(validation_accuracy)

            LOGGER.debug('Saving...')

            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                torch.save(network, str(model_path))
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

            if not clargs.constant_learning_rate:
                scheduler.step(1.0 - validation_accuracy)

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
    validation_data_loader = data.DataLoader(validation_data, batch_size=clargs.validation_batch_size,
                                             num_workers=clargs.num_workers)

    validation_loss_function = functools.partial(functional.cross_entropy, size_average=False)

    network = torch.load(clargs.model_path)
    network.cuda()

    validation_loss, validation_accuracy = training.evaluate_loss_and_accuracy(
        validation_data_loader, network, validation_loss_function, cuda=True)

    LOGGER.info('validation loss:       {}\n'
                'validation accuracy:   {}'
                .format(validation_loss, validation_accuracy))


def get_args(raw_args: typing.List[str]):
    arg_parser = argparse.ArgumentParser(
        description='Trains Resnet from the PyTorch models on the iMaterialist training data, and evaluates trained '
                    'resnet models on iMaterialist validation data.')
    arg_parser.add_argument(
        '--verbose', '-v', action='count', default=0,
        help='Indicate the level of verbosity. Include once to get info logs and twice to include debug-level logs.')
    arg_parser.add_argument('--num-workers', '-w', type=int, default=8,
                            help='Number of processes concurrently loading images from the training dataset.')

    subparsers = arg_parser.add_subparsers(dest='subparser_name')

    train_subcommand_parser = subparsers.add_parser(
        'train', help='Trains a Resnet model on the iMaterialist training data.')
    train_subcommand_parser.add_argument('--epoch-limit', '-l', type=int, default=41,
                                         help='The maximum number of epochs for which the network will train.')
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
    train_subcommand_parser.add_argument('--constant-learning-rate', '-c', action='store_true', default=False,
                                         help='Never decrease the learning rate during training.')
    train_subcommand_parser.add_argument('--training-batch-size', '-b', type=int, default=64,
                                         help='Training batch size.')
    train_subcommand_parser.add_argument('--validation-batch-size', '-a', type=int, default=8,
                                         help='Validation batch size. This parameter is purely for memory management '
                                              'purposes. A smaller validation batch size reduces the amount of memory '
                                              'used for validation image loading and computation on the GPU, '
                                              'leaving more memory for a larger training batch size.')
    train_subcommand_parser.set_defaults(train=True)

    evaluate_subcommand_parser = subparsers.add_parser(
        'evaluate', help='Evaluates a given Resnet model on the iMaterialist validation data.')
    evaluate_subcommand_parser.add_argument('--normalization', '-n', default='imaterialist',
                                            help='Normalization to use for evaluation.')
    evaluate_subcommand_parser.add_argument('model_directory',
                                            help='The directory containing the model to be evaluated.')
    evaluate_subcommand_parser.add_argument('--validation-batch-size', '-a', type=int, default=64,
                                            help='Validation batch size.')
    evaluate_subcommand_parser.set_defaults(evaluate=True)

    parsed_args = arg_parser.parse_args(args=raw_args)

    # Check additional restrictions on values and types.

    if parsed_args.num_workers < 1:
        arg_parser.error("--num-workers must be positive.")

    if parsed_args.train:
        if parsed_args.epoch_limit < 1:
            arg_parser.error("--epoch-limit must be positive.")
        if parsed_args.training_subset < 1:
            arg_parser.error("--training-subset must be positive.")
        if parsed_args.learning_rate <= 0.0:
            arg_parser.error("--learning-rate must be positive.")
        save_path = pathlib.Path(parsed_args.save_path)
        if save_path.exists():
            arg_parser.error("--save-path must not already exist.")
        if parsed_args.training_batch_size < 1:
            arg_parser.error("--training-batch-size must be positive.")
        if parsed_args.validation_batch_size < 1:
            arg_parser.error("--validation-batch-size must be positive.")

    elif parsed_args.evaluate:
        load_path = pathlib.Path(parsed_args.model_directory)
        if not load_path.exists():
            arg_parser.error("model_directory must exist.")
        elif not load_path.is_dir():
            arg_parser.error("model_directory must be a directory.")

    return parsed_args


def main():

    parsed_args = get_args(sys.argv[1:])

    if parsed_args.verbose == 0:
        LOGGER.setLevel(logging.WARNING)
    elif parsed_args.verbose == 1:
        LOGGER.setLevel(logging.INFO)
    elif parsed_args.verbose >= 2:
        LOGGER.setLevel(logging.DEBUG)

    if parsed_args.subparser_name == 'train':
        train(parsed_args)
    elif parsed_args.subparser_name == 'evaluate':
        evaluate(parsed_args)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [%(name)s] %(levelname)s %(message)s', level=logging.DEBUG)
    main()
