from yacs.config import CfgNode as CN

_C = CN()
_C.PATH = CN()
# output dir
_C.PATH.OUTPUT_DIR = '/home1/gmf/dzc/experiments/gait_recognition'
# experiment name
_C.PATH.EXPERIMENT_DIR = 'experiment'
# checkpoint folder name
_C.PATH.CHECKPOINT_DIR = 'checkpoints'
# folder for containing logs
_C.PATH.LOG_STORE_DIR = 'logs'
# path for containing split record
_C.PATH.SPLIT_RECORD_DIR = 'partition'
# model settings
_C.MODEL = CN()
_C.MODEL.NAME = 'SetNet'
_C.MODEL.NUM_FEATURES = 256
_C.MODEL.ACTIVATION = 'leaky_relu'
_C.MODEL.BNNECK = False
_C.MODEL.DEVICE = 'cuda'
_C.MODEL.DEVICE_ID = '2,'
_C.MODEL.CE_DIVIDED = 10
# dataset settings
_C.DATASET = CN()
_C.DATASET.DATASET_DIR = '/home1/gmf/dzc/GaitDatasetB-silh-processed'
_C.DATASET.NUM_WORKERS = 8
_C.DATASET.RESOLUTION = 64
_C.DATASET.BOUNDARY = 73
_C.DATASET.SHUFFLE = False
# logger settings
_C.LOGGER = CN()
_C.LOGGER.WRITE_TO_FILE = True
_C.LOGGER.LEVEL = 'DEBUG'
_C.LOGGER.FORMAT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
_C.LOGGER.NAME = 'logger'
# train settings
_C.TRAIN = CN()
_C.TRAIN.RECORD_STEP = 5000
_C.TRAIN.DISPLAY_INFO_STEP = 2500
_C.TRAIN.BATCH_SIZE = (8, 16)
_C.TRAIN.RESTORE_FROM_ITER = 0
_C.TRAIN.MAX_ITERS = 130000
_C.TRAIN.FRAME_NUM = 30
_C.TRAIN.CACHE = True
_C.TRAIN.USE_SCHEDULER = True
# triplet loss settings
_C.TRIPLET_LOSS = CN()
_C.TRIPLET_LOSS.TYPE = 'full'
_C.TRIPLET_LOSS.MARGIN = 0.2
# test settings
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 16
_C.TEST.CACHE = False
_C.TEST.TEST_ITER = 1000
_C.TEST.NUM_RANKS = 5
# solver
_C.SOLVER = CN()
_C.SOLVER.BASE_LR = 1e-4
_C.SOLVER.BIAS_LR_FACTOR = 2
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.
_C.SOLVER.OPTIMIZER_NAME = 'Adam'
_C.SOLVER.OPTIMIZER_MANNER = 'layer-wise'
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.MILESTONES = [40000, 70000, 100000]
_C.SOLVER.WARMUP_ITERS = 1000