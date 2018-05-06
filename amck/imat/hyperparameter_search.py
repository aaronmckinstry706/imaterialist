import logging
import sys


import amck.imat.finetune_resnet as finetune_resnet


LOGGER = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
LOGGER.addHandler(stream_handler)
LOGGER.setLevel(logging.DEBUG)


def main():
    for lr in [1.0 / (2**i) for i in range(13)]:
        args = ('-vv '
                '--num-workers 8 '
                'train '
                '--pretrained '
                '--training-subset 16000 '
                '--epoch-limit 2 '
                '--training-batch-size 32 '
                '--validation-batch-size 16 '
                '--learning-rate ' + str(lr) + ' '
                '--save-path models/densenet121/hypersearch_subset16000_lr' + str(lr)).split(' ')
        clargs = finetune_resnet.get_args(args)
        finetune_resnet.train(clargs)


if __name__ == '__main__':
    main()
