from yacs.config import CfgNode as CN

_C = CN()

# ===================== MODEL CONFIGURATION =====================
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"  # Device to use for training (options: 'cuda' or 'cpu')
_C.MODEL.DEVICE_ID = '0'  # GPU ID to use
_C.MODEL.NAME = 'STMI'  # Name of the model
_C.MODEL.PRETRAIN_PATH_T = '/path/to/your/vitb_16_224_21k.pth'  # Path to the pretrained ViT backbone
_C.MODEL.NECK = 'bnneck'  # Whether to use BNNeck (options: 'bnneck' or 'no')
_C.MODEL.IF_WITH_CENTER = 'no'  # Whether to include center loss in training (options: 'yes' or 'no')
_C.MODEL.ID_LOSS_TYPE = 'softmax'  # Type of ID loss
_C.MODEL.METRIC_LOSS_TYPE = 'triplet'  # Type of metric loss (options: 'triplet', 'center', 'triplet_center')
_C.MODEL.ID_LOSS_WEIGHT = 0.25  # Weight for ID loss，0.25 for CLIP, 1.0 for ViT
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0  # Weight for triplet loss
_C.MODEL.DIST_TRAIN = False  # Whether to train with multi-GPU DDP mode (options: True or False)
_C.MODEL.IF_LABELSMOOTH = 'on'  # Whether to use label smoothing (options: 'on' or 'off')

# Borrowed from MambaPro and IDEA
_C.MODEL.PROMPT = False  # Whether to enable prompt tuning
_C.MODEL.ADAPTER = False  # Whether to enable adapter tuning
_C.MODEL.FROZEN = False  # Whether to freeze the backbone
_C.MODEL.PREFIX = False  # Whether to use modal prefixes in dataloader processing
_C.MODEL.TEXT_PROMPT = 0  # Number of learnable text prompts in InverseNet
_C.MODEL.INVERSE = False  # Whether to use the inverse network in IMFE
_C.MODEL.DA = False  # Whether to use deformable aggregation
_C.MODEL.DA_SHARE = False  # Whether to share offsets across modalities
_C.MODEL.OFF_FAC = 5.0  # Offset factor to control offset magnitude

# Transformer settings
_C.MODEL.DROP_PATH = 0.1  # DropPath rate
_C.MODEL.DROP_OUT = 0.0  # Dropout rate
_C.MODEL.ATT_DROP_RATE = 0.0  # Attention dropout rate
_C.MODEL.TRANSFORMER_TYPE = 'vit_base_patch16_224'  # Type of transformer
_C.MODEL.STRIDE_SIZE = [16, 16]  # Stride size for feature extraction
_C.MODEL.FORWARD = 'old'  # Forward function type in the backbone (options: 'cls return' or 'patch return')
_C.MODEL.DIRECT = 1  # Whether to use contact features

# SIE parameters
_C.MODEL.SIE_COE = 1.0  # Coefficient for SIE
_C.MODEL.SIE_CAMERA = True  # Whether to use camera information
_C.MODEL.SIE_VIEW = False  # Whether to use view information (not used)

# STMI-specific configurations
_C.MODEL.MASK = True  # Whether to use SFM   
_C.MODEL.PROBABILITY = 0.0  # The random probability in SFM
_C.MODEL.STR_CHI = True  # Whether to use STR and CHI
_C.MODEL.LEARNABLE_TOKENS = 4  # The number of learnable tokens in STR
_C.MODEL.THRESHOLD = 5.0  # The threshold in CHI 

# ===================== INPUT CONFIGURATION =====================
_C.INPUT = CN()
_C.INPUT.SIZE_TRAIN = [256, 128]  # Image size during training
_C.INPUT.SIZE_TEST = [256, 128]  # Image size during testing
_C.INPUT.PROB = 0.5  # Probability for random horizontal flip
_C.INPUT.RE_PROB = 0.5  # Probability for random erasing
_C.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]  # Mean values for image normalization
_C.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]  # Standard deviation values for image normalization
_C.INPUT.PADDING = 10  # Padding size for images

# ===================== DATASET CONFIGURATION =====================
_C.DATASETS = CN()
_C.DATASETS.NAMES = ('RGBNT201')  # Names of datasets for training
_C.DATASETS.ROOT_DIR = './data'  # Root directory for datasets

# ===================== DATALOADER CONFIGURATION =====================
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 14  # Number of data loading threads
_C.DATALOADER.SAMPLER = 'softmax_triplet'  # Sampler for data loading
_C.DATALOADER.NUM_INSTANCE = 8  # Number of instances per batch

# ===================== SOLVER CONFIGURATION =====================
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "Adam"  # Name of optimizer
_C.SOLVER.MAX_EPOCHS = 50  # Maximum number of training epochs
_C.SOLVER.BASE_LR = 0.00035  # Base learning rate
_C.SOLVER.LARGE_FC_LR = False  # Whether to use a larger learning rate for fully connected layers
_C.SOLVER.BIAS_LR_FACTOR = 2  # Learning rate factor for bias terms
_C.SOLVER.MOMENTUM = 0.9  # Momentum for optimizer
_C.SOLVER.MARGIN = 0.3  # Margin for triplet loss
_C.SOLVER.CLUSTER_MARGIN = 0.3  # Margin for cluster loss
_C.SOLVER.CENTER_LR = 0.5  # Learning rate for center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005  # Weight for center loss
_C.SOLVER.RANGE_K = 2  # K value for range loss
_C.SOLVER.RANGE_MARGIN = 0.3  # Margin for range loss
_C.SOLVER.RANGE_ALPHA = 0  # Alpha value for range loss
_C.SOLVER.RANGE_BETA = 1  # Beta value for range loss
_C.SOLVER.RANGE_LOSS_WEIGHT = 1  # Weight for range loss
_C.SOLVER.WEIGHT_DECAY = 0.0001  # Weight decay for optimizer
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0001  # Weight decay for bias terms
_C.SOLVER.GAMMA = 0.1  # Decay rate for learning rate
_C.SOLVER.STEPS = (40, 70)  # Steps for learning rate decay
_C.SOLVER.WARMUP_FACTOR = 0.01  # Warmup factor for learning rate
_C.SOLVER.WARMUP_ITERS = 10  # Number of warmup iterations
_C.SOLVER.WARMUP_METHOD = "linear"  # Warmup method (options: 'constant', 'linear')
_C.SOLVER.COSINE_MARGIN = 0.5  # Margin for cosine loss
_C.SOLVER.COSINE_SCALE = 30  # Scale for cosine loss
_C.SOLVER.SEED = 1111  # Random seed for reproducibility
_C.MODEL.NO_MARGIN = True  # Whether to disable margin
_C.SOLVER.CHECKPOINT_PERIOD = 50  # Period for saving checkpoints
_C.SOLVER.LOG_PERIOD = 10  # Period for logging training progress
_C.SOLVER.EVAL_PERIOD = 1  # Period for evaluation
_C.SOLVER.IMS_PER_BATCH = 64  # Number of images per batch

# ===================== TEST CONFIGURATION =====================
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 128  # Number of images per batch during testing
_C.TEST.RE_RANKING = 'no'  # Whether to use re-ranking (options: 'yes', 'no')
_C.TEST.WEIGHT = ""  # Path to the trained model weights
_C.TEST.NECK_FEAT = 'before'  # Which BNNeck feature to use for testing (options: 'before' or 'after')
_C.TEST.FEAT_NORM = 'yes'  # Whether to normalize features before testing
_C.TEST.MISS = 'None'  # Modality missing pattern (options: 'None', 'r', 'n', 't', 'rn', 'rt', 'nt')

# ===================== MISC OPTIONS =====================
_C.OUTPUT_DIR = "./STMI"  # Output directory for checkpoints and logs