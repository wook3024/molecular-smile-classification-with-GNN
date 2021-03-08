''' Default '''
RADIUS = 1
LAYER_HIDDEN = 2
LAYER_OUTPUT = 3
BATCH_TRAIN = 32
BATCH_TEST = 32
LR = 1e-4
LR_DECAY = 0.99
DECAY_INTERVAL = 10
ITERATION = 100
MODE = 'train'
LOSS_TYPE = 'FocalLoss'
LOAD_MODEL_PATH = ''
MODEL_TYPE = 'GCN'
DATASET = 'bionsight'
HIDDEN_DIM = 128
OUT_DIM = 64
INPUT_DIM = 512
NORM_TYPE = 'gn'
SKIP_CONNECTION_TYPE = 'gsc'
USE_AUGMENTATION = False

''' File '''
CHECKPOINT = 'checkpoint.pth'
fn_LOGGER = 'logger.log'