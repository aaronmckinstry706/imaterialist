import argparse
import collections
import multiprocessing
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
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms


import amck.imat.stopwatch as stopwatch


arg_parser = argparse.ArgumentParser(description='Trains a pre-trained Resnet-18 from the PyTorch models and fine-tunes'
                                                 ' the model on the iMaterialist training data.')
arg_parser.add_argument('--num-workers', '-w', type=int, help='Number of processes concurrently loading images from '
                                                              'the training dataset.')
arg_parser.add_argument('--limit', '-l', type=int, default=1000000, help='The maximum number of descent steps for '
                                                                         'which the network will train.')
arg_parser.add_argument('--short-history', '-s', type=int, default=500, help='The length of the short history to  '
                                                                             'display in the plot of metrics.')
arg_parser.add_argument('--batch-size', '-b', type=int, default=128, help='Batch size.')
arg_parser.add_argument('--save-interval', '-i', type=int, default=500, help='Every SAVE_INTERVAL iterations, '
                                                                             'a copy of the model will be saved.')
parsed_args = arg_parser.parse_args()

command_queue = queue.Queue()
metrics = collections.defaultdict(list)
plotting_process = None


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
            plotting_process = multiprocessing.Process(target=metrics_plotter()).start()


def data_generator(dataset: data.Dataset, shuffle=True):
    while True:
        data_loader = data.DataLoader(dataset, batch_size=parsed_args.batch_size, shuffle=shuffle,
                                      num_workers=parsed_args.num_workers)
        for x in data_loader:
            yield x


# We will use the normalization transform on the images, described at
# http://pytorch.org/docs/0.3.0/torchvision/models.html.
image_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

training_data = datasets.ImageFolder('data/training', transform=image_transform)

pretrained_resnet: models.ResNet = models.resnet18(pretrained=True)
pretrained_resnet.fc = nn.Linear(512, 128)
pretrained_resnet.cuda()
optimizer = optim.Adam(pretrained_resnet.fc.parameters())
scheduler = lr_scheduler.MultiStepLR(optimizer, [100000], gamma=0.1)

command_thread = threading.Thread(target=command_listener)
command_thread.daemon = True
command_thread.start()
stopwatches: typing.DefaultDict[str, stopwatch.Stopwatch] = collections.defaultdict(stopwatch.Stopwatch)

stopwatches['total_duration'].start()
stopwatches['duration'].start()
for i, (images, labels) in enumerate(data_generator(training_data)):
    images = autograd.Variable(images).cuda()
    labels = autograd.Variable(labels).cuda()
    loss = functional.cross_entropy(pretrained_resnet(images), labels)
    pretrained_resnet.zero_grad()
    loss.backward()
    optimizer.step()
    metrics['loss'].append(loss.cpu().data.numpy()[0])
    stopwatches['duration'].lap()
    current_iteration_time = stopwatches['duration'].lap_times()[-1]
    if i % parsed_args.save_interval == 0:
        torch.save(pretrained_resnet, 'models/finetuned_resnet')
    print('iteration={}, loss={}, time={}'.format(i, loss.cpu().data.numpy()[0], current_iteration_time))
    if parsed_args.limit == i:
        break
    if not command_queue.empty():
        command = command_queue.get()
        if command == 'exit':
            break
stopwatches['duration'].lap()
stopwatches['duration'].stop()
stopwatches['total_duration'].lap()
stopwatches['total_duration'].stop()
print('total_time={}'.format(stopwatches['total_duration'].lap_times()[-1]))
torch.save(pretrained_resnet, 'models/finetuned_resnet')