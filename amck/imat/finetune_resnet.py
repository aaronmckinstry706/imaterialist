import argparse
import collections
import datetime
import multiprocessing
import random
import sys
import threading
import typing
import queue


import matplotlib.pyplot as pyplot
import torch
import torch.autograd as autograd
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


import amck.imat.stopwatch as stopwatch


START_TIME_STR = str(datetime.datetime.now()).replace(' ', '_')

arg_parser = argparse.ArgumentParser(description='Trains Resnet-18 from the PyTorch models on the iMaterialist training'
                                                 ' data.')
arg_parser.add_argument('--num-workers', '-w', type=int, default=8, help='Number of processes concurrently loading '
                                                                         'images from the training dataset.')
arg_parser.add_argument('--step-limit', '-l', type=int, default=640000, help='The maximum number of descent steps for '
                                                                             'which the network will train.')
arg_parser.add_argument('--short-history', '-s', type=int, default=500, help='The length of the short history to  '
                                                                             'display in the plot of metrics.')
arg_parser.add_argument('--batch-size', '-b', type=int, default=256, help='Batch size.')
arg_parser.add_argument('--save-interval', '-i', type=int, default=500, help='Every SAVE_INTERVAL iterations, '
                                                                             'a copy of the model will be saved.')
arg_parser.add_argument('--pretrained', '-p', action='store_true', default=False, help='Indicates that the model '
                                                                                       'should be pretrained on '
                                                                                       'ImageNet.')
arg_parser.add_argument('--training-subset', '-t', type=int, default=sys.maxsize, help='Indicates that only the first '
                                                                                       'TRAINING_SUBSET number of '
                                                                                       'samples should be used as '
                                                                                       'data. (This is for debugging.)')
arg_parser.add_argument('--learning-rate', '-r', type=float, default=0.001, help='Initial learning rate.')
parsed_args = arg_parser.parse_args()

command_queue = queue.Queue()
metrics = collections.defaultdict(list)


def metrics_plotter():
    def plot_metrics():
        pyplot.figure()
        for p, (name, vals) in enumerate(metrics.items()):
            if len(vals) > 1:
                pyplot.ion()
                pyplot.subplot(len(metrics), 2, 2 * p + 1)
                pyplot.plot(vals)
                pyplot.subplot(len(metrics), 2, 2 * p + 2)
                pyplot.plot(vals[-parsed_args.short_history:])
                pyplot.show(block=True)
    return plot_metrics


def command_listener():
    while True:
        command = input()
        command_queue.put(command)
        if command == 'metrics':
            multiprocessing.Process(target=metrics_plotter()).start()


def data_generator(dataset: data.Dataset, shuffle=True):
    if parsed_args.training_subset < len(dataset):
        subset_indices = random.sample([i for i in range(len(dataset))], parsed_args.training_subset)
        data_sampler = sampler.SubsetRandomSampler(subset_indices)
    else:
        data_sampler = sampler.RandomSampler(dataset)
    data_loader = data.DataLoader(dataset, batch_size=parsed_args.batch_size, num_workers=parsed_args.num_workers,
                                  sampler=data_sampler)
    while True:
        num_images_so_far = 0
        for (images, labels) in data_loader:
            yield (images, labels)
            num_images_so_far += len(images)
            if num_images_so_far >= parsed_args.training_subset:
                print('Finished epoch through dataset.')
                break


image_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.6837, 0.6461, 0.6158], std=[0.2970, 0.3102, 0.3271])
])

training_data = datasets.ImageFolder('data/training', transform=image_transform)

resnet: models.ResNet = models.resnet18(pretrained=parsed_args.pretrained)
resnet.fc = nn.Linear(512, 128)
resnet.cuda()
optimizer = optim.Adam(resnet.fc.parameters(), lr=parsed_args.learning_rate, weight_decay=0.0001)

command_thread = threading.Thread(target=command_listener)
command_thread.daemon = True
command_thread.start()
stopwatches: typing.DefaultDict[str, stopwatch.Stopwatch] = collections.defaultdict(stopwatch.Stopwatch)

stopwatches['total_duration'].start()
stopwatches['duration'].start()
for i, (images, labels) in enumerate(data_generator(training_data)):
    images = autograd.Variable(images).cuda()
    labels = autograd.Variable(labels).cuda()
    loss = functional.cross_entropy(resnet(images), labels)
    resnet.zero_grad()
    loss.backward()
    optimizer.step()
    metrics['loss'].append(loss.cpu().data.numpy()[0])
    stopwatches['duration'].lap()
    current_iteration_time = stopwatches['duration'].lap_times()[-1]
    if i % parsed_args.save_interval == 0:
        torch.save(resnet, 'models/finetuned_resnet_{}'.format(START_TIME_STR))
    print('iteration={}, loss={}, time={}'.format(i, loss.cpu().data.numpy()[0], current_iteration_time))
    if parsed_args.step_limit == i:
        break
    if not command_queue.empty():
        command = command_queue.get()
        if command == 'exit':
            break
        elif command == 'images':
            image_grid: torch.FloatTensor = utils.make_grid(images.data[:64, :, :, :], nrow=8, padding=5)
            print(image_grid.shape)
            pyplot.figure()
            pyplot.imshow(image_grid.permute(1, 2, 0))
            pyplot.show(block=True)
stopwatches['duration'].lap()
stopwatches['duration'].stop()
stopwatches['total_duration'].lap()
stopwatches['total_duration'].stop()
print('total_time={}'.format(stopwatches['total_duration'].lap_times()[-1]))
torch.save(resnet, 'models/finetuned_resnet')
