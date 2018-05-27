import argparse
import collections
import datetime
import functools
import itertools
import logging
import pathlib
import random
import sys
import typing


import matplotlib.pyplot as pyplot
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
import torchvision.utils as utils


import amck.imat.model_average as model_average
import amck.imat.stopwatch as stopwatch
import amck.imat.training as training


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
        # transforms.ColorJitter(brightness=0.2, contrast=0.8, saturation=0.8, hue=0.3),
        # transforms.RandomGrayscale(p=0.1),
        transforms.Resize(224),
        transforms.RandomCrop(224),
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

    network: models.VGG = models.vgg19_bn(pretrained=clargs.pretrained)
    network.classifier[6] = nn.Linear(network.classifier[6].in_features, 128)
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
                epoch_loss_history = [batch_loss / clargs.training_batch_size for batch_loss in epoch_loss_history]
                training_stopwatch.lap()

            LOGGER.debug('Validating...')

            with validation_stopwatch:
                validation_loss, validation_accuracy = training.evaluate_loss_and_accuracy(
                    validation_data_loader, network, validation_loss_function, cuda=True,
                    progress_bar=(clargs.verbose >= 2))
                validation_loss /= len(validation_data)
                validation_stopwatch.lap()

            metrics['training_loss'].extend(epoch_loss_history)
            metrics['validation_loss'].append(validation_loss)
            metrics['training_accuracy'].extend(epoch_acccuracy_history)
            metrics['validation_accuracy'].append(validation_accuracy)

            LOGGER.debug('Saving...')

            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                torch.save(network, str(model_path))
                saving_stopwatch.lap()

            torch.save(metrics, metrics_path)

            epoch_stopwatch.lap()

            print('epoch {epoch}\n'
                  '- total duration:                  {total_duration}\n'
                  '- validation loss:                 {validation_loss}\n'
                  '- validation accuracy:             {validation_accuracy}\n'
                  '- average training-batch loss:     {avg_training_loss}\n'
                  '- average training-batch accuracy: {avg_training_accuracy}'
                  .format(epoch=i,
                          total_duration=epoch_stopwatch.lap_times()[-1],
                          validation_loss=validation_loss,
                          validation_accuracy=validation_accuracy,
                          avg_training_loss=sum(epoch_loss_history)/len(epoch_loss_history),
                          avg_training_accuracy=sum(epoch_acccuracy_history)/len(epoch_acccuracy_history)))
            print('- training duration:              {training_duration}\n'
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
        inverse_normalize = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])])
    elif clargs.normalization == 'imaterialist':
        # From iMaterialist.
        normalize = transforms.Normalize(mean=[0.6837, 0.6461, 0.6158], std=[0.2970, 0.3102, 0.3271])
        inverse_normalize = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.2970, 1 / 0.3102, 1 / 0.3271]),
            transforms.Normalize(mean=[-0.6837, -0.6461, -0.6158], std=[1., 1., 1.])])

    image_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    validation_data = datasets.ImageFolder('data/validation', transform=image_transform)
    validation_data_loader = data.DataLoader(validation_data, batch_size=clargs.validation_batch_size,
                                             num_workers=clargs.num_workers)

    validation_loss_function = functools.partial(functional.nll_loss, size_average=False)

    loaded_models: typing.List[nn.Module] = []
    for model_directory in clargs.model_directories:
        current_model = torch.load(str(pathlib.Path(model_directory) / 'model'))
        loaded_models.append(current_model)
    network = model_average.ModelAverage(*loaded_models)
    network.cuda()

    validation_loss, validation_accuracy, error_samples = training.evaluate_and_get_error_sample(
        validation_data_loader, network, validation_loss_function, cuda=True, progress_bar=True)

    print('validation loss:       {}\n'
          'validation accuracy:   {}'
          .format(validation_loss, validation_accuracy))

    # Visualize the distribution of errors over each class.
    error_images, error_labels = zip(*error_samples)
    error_images = [inverse_normalize(image) for image in error_images]
    error_distribution_by_class = collections.defaultdict(float)
    for label in error_labels:
        error_distribution_by_class[str(label)] += 1
    # for key in error_labels:
    #     error_distribution_by_class[str(key)] /= len(error_labels)
    sorted_error_label_strings, sorted_error_label_distribution = zip(
        *sorted(error_distribution_by_class.items(), key=lambda t: (-t[1], t[0])))
    _, axes = pyplot.subplots(2, 1)
    pyplot.title('Error Distribution Over Classes')
    axes[0].bar(sorted_error_label_strings, sorted_error_label_distribution)
    axes[0].set_title('Counts')
    axes[1].bar(sorted_error_label_strings, [s for s in itertools.accumulate(sorted_error_label_distribution)])
    axes[1].set_title('Cumulative Counts')

    # Visualize a random subset of images from the set of errors.
    pyplot.figure()
    error_image_subset_indexes = random.sample(range(len(error_images)), 64)
    error_image_grid = utils.make_grid(
        [error_images[i] for i in error_image_subset_indexes],
        nrow=min(8, len(error_image_subset_indexes)))
    pyplot.imshow(error_image_grid.permute(1, 2, 0).numpy())
    for grid_index, error_index in enumerate(error_image_subset_indexes):
        if grid_index % 8 == 0 and grid_index > 0:
            print(end='\n')
        print(error_labels[error_index], end=' ')

    pyplot.show()


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
    evaluate_subcommand_parser.add_argument('model_directories', nargs='+',
                                            help='directories containing models to be averaged and evaluated.')
    evaluate_subcommand_parser.add_argument('--validation-batch-size', '-a', type=int, default=2,
                                            help='Validation batch size.')
    evaluate_subcommand_parser.set_defaults(evaluate=True)

    parsed_args = arg_parser.parse_args(args=raw_args)

    # Check additional restrictions on values and types.

    if parsed_args.num_workers < 1:
        arg_parser.error("--num-workers must be positive.")

    if parsed_args.subparser_name == 'train':
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

    elif parsed_args.subparser_name == 'evaluate':
        for model_directory in parsed_args.model_directories:
            load_path = pathlib.Path(model_directory)
            if not load_path.exists():
                arg_parser.error("model_directory must exist.")
            elif not load_path.is_dir():
                arg_parser.error("model_directory must be a directory.")

    return parsed_args


def main():

    parsed_args = get_args(sys.argv[1:])

    if parsed_args.verbose == 0:
        logging_level = logging.WARNING
    elif parsed_args.verbose == 1:
        logging_level = logging.INFO
    elif parsed_args.verbose >= 2:
        logging_level = logging.DEBUG
    logging.basicConfig(format='%(asctime)s [%(name)s] %(levelname)s %(message)s', level=logging_level)

    if parsed_args.subparser_name == 'train':
        train(parsed_args)
    elif parsed_args.subparser_name == 'evaluate':
        evaluate(parsed_args)


if __name__ == '__main__':
    main()
