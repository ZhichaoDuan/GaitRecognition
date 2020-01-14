from yacs.config import CfgNode as CN

_C = CN()
_C.PATH = CN()
# output dir
_C.PATH.OUTPUT_DIR = '/home1/gmf/dzc/experiments/gait_recognition'
# experiment name
_C.PATH.EXPERIMENT_DIR = 'test'
# checkpoint folder name
_C.PATH.CHECKPOINT_DIR = 'checkpoints'
# folder for containing logs
_C.PATH.LOG_STORE_DIR = 'logs'
# model settings
_C.MODEL = CN()
_C.MODEL.NAME = 'SetNet'
_C.MODEL.FILE_MIDDLE = 'test'
_C.MODEL.NUM_FEATURES = 256
_C.MODEL.BNNECK = False
_C.MODEL.DEVICE = 'cuda'
_C.MODEL.DEVICE_ID = '2'
_C.MODEL.N_SAVED = 5
# dataset settings
_C.DATASET = CN()
_C.DATASET.DATASET_DIR = '/home1/gmf/dzc/GaitDatasetB-silh-processed'
_C.DATASET.NUM_WORKERS = 4
_C.DATASET.RESOLUTION = 64
_C.DATASET.BOUNDARY = 73
# logger settings
_C.LOGGER = CN()
_C.LOGGER.WRITE_TO_FILE = True
_C.LOGGER.LEVEL = 'DEBUG'
_C.LOGGER.FORMAT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
_C.LOGGER.NAME = 'logger'
# train settings
_C.TRAIN = CN()
_C.TRAIN.RECORD_STEP = 2000
_C.TRAIN.DISPLAY_INFO_STEP = 1000
_C.TRAIN.BATCH_SIZE = (8, 16)
_C.TRAIN.RESTORE_FROM_ITER = 0
_C.TRAIN.MAX_ITERS = 120000
_C.TRAIN.FRAME_NUM = 30
_C.TRAIN.CACHE = False
# has to be one of 'normal', 'smooth' or 'none'
_C.TRAIN.CLS_LOSS_TYPE = 'normal'
_C.TRAIN.CE_DIVIDED = 10
_C.TRAIN.SMOOTH_EPSILON = 0.1
# triplet loss settings
_C.TRIPLET_LOSS = CN()
# full or hard
_C.TRIPLET_LOSS.TYPE = 'full'
_C.TRIPLET_LOSS.MARGIN = 0.2
# val settings
_C.VAL = CN()
_C.VAL.BATCH_SIZE = 16
_C.VAL.CACHE = False
_C.VAL.K = 1
# solver
_C.SOLVER = CN()
_C.SOLVER.BASE_LR = 1e-4
_C.SOLVER.BIAS_LR_FACTOR = 1
_C.SOLVER.WEIGHT_DECAY = 0.0
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.
_C.SOLVER.OPTIMIZER_NAME = 'Adam'
_C.SOLVER.MILESTONES = []
_C.SOLVER.WARMUP_ITERS = -1
_C.SOLVER.GAMMA = 1