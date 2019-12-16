from yacs.config import CfgNode as CN

_C = CN()
# output dir
_C.OUTPUT_DIR = '/home/gmf/duanzhichao/experiments/gait_recognition'
# experiment name
_C.EXPERIMENT = 'test1_test1000iters_hard_differ_with_officialmodel_withdelete_more_arc_in_offi_model'
# checkpoint folder name
_C.CHECKPOINT_DIR = 'checkpoints'
# cuda env
_C.CUDA_VISIBLE_DEVICES = '7'
# dataset dir
_C.DATASET_DIR = '/home/gmf/duanzhichao/datasets/GaitDatasetB-silh-processed'
# threads to use
_C.NUM_WORKERS = 8
# record for spliting training and testing sets
_C.RECORD = 'partition'
# input settings
_C.INPUT = CN()
_C.INPUT.RESOLUTION = 64
_C.INPUT.BOUNDARY = 73
_C.INPUT.SHUFFLE = False
# logger of training
_C.LOGGER = CN()
_C.LOGGER.WRITE_TO_FILE = True
_C.LOGGER.STORE_DIR = 'logs'
_C.LOGGER.LEVEL = 'DEBUG'
_C.LOGGER.FORMAT = '%(asctime)s::%(name)s::%(levelname)s::%(message)s'
_C.LOGGER.NAME = 'test_train_process_tryout_for_hard'
# train settings
_C.TRAIN = CN()
_C.TRAIN.RECORD_STEP = 100
_C.TRAIN.LR = 1e-4
_C.TRAIN.BATCH_SIZE = (8, 16)
_C.TRAIN.RESTORE_FROM_ITER = 0
_C.TRAIN.MAX_ITERS = 1000
_C.TRAIN.FRAME_NUM = 30
_C.TRAIN.CACHE = True
_C.TRAIN.TRIPLET_LOSS = CN()
_C.TRAIN.TRIPLET_LOSS.TYPE = 'hard'
_C.TRAIN.TRIPLET_LOSS.MARGIN = 0.2
# model settings
_C.MODEL = CN()
_C.MODEL.NAME = 'SetNet'
_C.MODEL.NUM_FEATURES = 256
_C.MODEL.ACTIVATION = 'leaky_relu'