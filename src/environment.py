import constant
import importlib
import os


def setup(args):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    

    constant.RADIUS = args.radius
    constant.LAYER_HIDDEN = args.layer_hidden
    constant.LAYER_OUTPUT = args.layer_output
    constant.BATCH_TRAIN = args.batch_train
    constant.BATCH_TEST = args.batch_test
    constant.LR = args.lr
    constant.LR_DECAY = args.lr_decay
    constant.DECAY_INTERVAL = args.decay_interval
    constant.ITERATION = args.iteration
    constant.MODEL_TYPE = args.model_type
    constant.MODE = args.mode
    constant.LOSS_TYPE = args.loss_type
    constant.LOAD_MODEL_PATH = args.load_model_path
    constant.DATASET = args.dataset
    constant.HIDDEN_DIM = args.hidden_dim
    constant.OUT_DIM = args.out_dim
    constant.INPUT_DIM = args.input_dim
    constant.NORM_TYPE = args.norm_type
    constant.SKIP_CONNECTION_TYPE = args.skip_connection_type
    constant.USE_AUGMENTATION = args.use_augmentation

    constant.CHECKPOINT = (f"{constant.MODEL_TYPE}_{constant.DATASET}")
    constant.LOGGER = (f"{constant.MODEL_TYPE}_{constant.DATASET}")
    constant.ROOT_DIR = f'../output/{constant.CHECKPOINT}-{args.loss_type}-{args.norm_type}-{args.skip_connection_type}-{args.input_dim}-{args.hidden_dim}-{args.out_dim}-{args.layer_hidden}-{args.layer_output}-{args.batch_train}'

    print('='*110)
    print("MODEL:", constant.MODEL_TYPE)
    print("DATASET:", constant.DATASET)
    print("SAVE_DIR:", constant.ROOT_DIR)