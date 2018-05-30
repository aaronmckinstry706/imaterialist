# Parts of code taken from https://www.kaggle.com/aloisiodn/python-3-download-multi-proc-prog-bar-resume by Dourado.
# Improvements on the original script:
#   * you can choose which dataset to download;
#   * uses threads instead of processes;
#   * unpacks data into .../label/id.jpg directory structure, which can be used easily via classes in PyTorch;
#   * performance-relevant parameters are command line arguments.
# For performance parameters, the recommended values (from my machine; probably requires tweaking for others) are 100
# connection pools, 128 threads. Not all images with working URLs will be retrieved, but about 90-95% of them will. As
# a consequence, to ensure that nearly all images have been downloaded, repeat the script 3-4 times.

import argparse
import io
import json
import logging
import multiprocessing.pool as pool
import pathlib
import random
import sys
import typing
import urllib3


import PIL.Image as Image
from tqdm import tqdm


# Get command line arguments.

arg_parser = argparse.ArgumentParser(
    description='Downloads the data files using the links given in the JSON training, validation, and test files. '
                'Assumes that the files are stored in the directory data/metadata (relative to the current working '
                'directory). Training files will be written to data/training/label_id/image_id.jpg, validation files '
                'will be written to data/validation/label_id/image_id.jpg, and test files will be written to '
                'data/testing/image_id.jpg.')
arg_parser.add_argument(
    '--num-pools', '-p', type=int, default=10, help='Number of connection pools to cache at one time.')
arg_parser.add_argument(
    '--num-workers', '-w', type=int, default=8, help='Number of threads to perform downloads.')
arg_parser.add_argument(
    '--verbose', '-v', action='count', help='Print additional output messages. Can be passed multiple times. Once '
                                            'prints additional status information, and two or more times prints '
                                            'debugging information.', default=0)
arg_parser.add_argument(
    '--limit', '-l', type=int, default=sys.maxsize, help='Maximum number of files to download before stopping.')
arg_parser.add_argument(
    '--re-download', action='store_true', default=False, help='Whether to re-download existing files.')
arg_parser.add_argument(
    '--dataset', '-d', type=str, choices={'training', 'validation', 'testing'}, help='Which dataset to download.')
parsed_args = arg_parser.parse_args()

# Set up logging.

urllib3.disable_warnings()
LOGGER = logging.getLogger(__name__)
STDOUT_HANDLER = logging.StreamHandler(sys.stdout)
if parsed_args.verbose == 1:
    STDOUT_HANDLER.setLevel(logging.INFO)
elif parsed_args.verbose >= 2:
    STDOUT_HANDLER.setLevel(logging.DEBUG)
LOGGER.addHandler(STDOUT_HANDLER)
LOGGER.setLevel(logging.DEBUG)

# Initialize globals.

failed_downloads = []
http = urllib3.PoolManager(num_pools=parsed_args.num_pools)


def download_image(url: str, filepath: pathlib.Path):
    global parsed_args
    global http

    file_exists = filepath.exists()
    if parsed_args.re_download and file_exists:
        filepath.unlink()
    elif not parsed_args.re_download and file_exists:
        return

    response = http.request('GET', url, timeout=urllib3.Timeout(10))
    image_data = response.data
    pil_image = Image.open(io.BytesIO(image_data))
    pil_image_rgb = pil_image.convert('RGB')
    pil_image_rgb.save(str(filepath), format='JPEG', quality=90)


def download_labeled_image(info: typing.Tuple[str, int, int, pathlib.Path]):
    global failed_downloads
    url: str = info[0]
    image_id: int = info[1]
    label_id: int = info[2]
    base_dir: pathlib.Path = info[3]
    label_dir = base_dir.joinpath(str(label_id))
    filepath = label_dir.joinpath(str(image_id) + '.jpg')

    label_dir.mkdir(parents=True, exist_ok=True)

    try:
        download_image(url, filepath)
    except Exception as e:
        failed_downloads.append((image_id, str(e)))


def download_unlabeled_image(info: typing.Tuple[str, int, pathlib.Path]):
    global failed_downloads

    url: str = info[0]
    image_id: int = info[1]
    base_dir: pathlib.Path = info[2]
    label_dir = base_dir.joinpath('dummy-class')
    filepath = label_dir.joinpath(str(image_id) + '.jpg')

    label_dir.mkdir(parents=True, exist_ok=True)

    try:
        download_image(url, filepath)
    except Exception as e:
        failed_downloads.append((image_id, str(e)))


training_base_dir = pathlib.Path('data/training')
validation_base_dir = pathlib.Path('data/validation')
testing_base_dir = pathlib.Path('data/testing')
metadata_base_dir = pathlib.Path('data/metadata')

with metadata_base_dir.joinpath('train.json').open('r') as training_urls_file:
    training_urls_json = json.load(training_urls_file)
with metadata_base_dir.joinpath('validation.json').open('r') as validation_urls_file:
    validation_urls_json = json.load(validation_urls_file)
with metadata_base_dir.joinpath('test.json').open('r') as testing_urls_file:
    testing_urls_json = json.load(testing_urls_file)

num_training_images = len(training_urls_json['images'])
num_validation_images = len(validation_urls_json['images'])
num_testing_images = len(testing_urls_json['images'])
LOGGER.info('{} training images, {} validation images, and {} testing images.'.format(
    num_training_images, num_validation_images, num_testing_images))

thread_pool = pool.ThreadPool(processes=parsed_args.num_workers)

if parsed_args.dataset == 'training':
    training_image_info = []
    for image_info, annotation_info in zip(training_urls_json['images'], training_urls_json['annotations']):
        training_image_info.append((image_info['url'][0], image_info['image_id'], annotation_info['label_id'],
                                    training_base_dir))
    random.shuffle(training_image_info)

    with tqdm(total=len(training_image_info), desc='Training images') as t:
        for i, _ in enumerate(thread_pool.imap_unordered(download_labeled_image, training_image_info)):
            t.update(1)
            if i >= parsed_args.limit:
                break

elif parsed_args.dataset == 'validation':
    validation_image_info = []
    for image_info, annotation_info in zip(validation_urls_json['images'], validation_urls_json['annotations']):
        validation_image_info.append((image_info['url'][0], image_info['image_id'], annotation_info['label_id'],
                                      validation_base_dir))
    random.shuffle(validation_image_info)

    with tqdm(total=len(validation_image_info), desc='Validation images') as t:
        for i, _ in enumerate(thread_pool.imap_unordered(download_labeled_image, validation_image_info)):
            t.update(1)
            if i >= parsed_args.limit:
                break

elif parsed_args.dataset == 'testing':
    testing_image_info = []
    for image_info in testing_urls_json['images']:
        testing_image_info.append((image_info['url'][0], image_info['image_id'], testing_base_dir))
    random.shuffle(testing_image_info)

    with tqdm(total=len(testing_image_info), desc='Testing images') as t:
        for i, _ in enumerate(thread_pool.imap_unordered(download_unlabeled_image, testing_image_info)):
            t.update(1)
            if i >= parsed_args.limit:
                break

LOGGER.info('{} images could not be retrieved.'.format(len(failed_downloads)))
