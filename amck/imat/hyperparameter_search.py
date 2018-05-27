import logging


import amck.imat.finetune_resnet as finetune_resnet


LOGGER = logging.getLogger(__name__)


def main():
    for lr in [1.0 / (2**(19 - i)) for i in range(20)]:
        args = ('-vv '
                '--num-workers 8 '
                'train '
                '--pretrained '
                '--training-subset 16000 '
                '--epoch-limit 2 '
                '--training-batch-size 16 '
                '--validation-batch-size 2 '
                '--learning-rate ' + str(lr) + ' '
                '--save-path models/vgg19_bn/hypersearch_subset16000augmented_lr' + str(lr)).split(' ')
        clargs = finetune_resnet.get_args(args)
        finetune_resnet.train(clargs)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [%(name)s] %(levelname)s %(message)s', level=logging.DEBUG)
    main()
