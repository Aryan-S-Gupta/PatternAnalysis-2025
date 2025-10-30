# Paths
DRIVE_ROOT = "/content/drive/MyDrive/siamese_project/data"       # project data root
# ISIC images directory
IMAGES_DIR = "/content/train-image/image"
META_CSV = "/content/train-metadata.csv"                         # metadata CSV path
ARTIFACTS = "/content/drive/MyDrive/siamese_project/outputs"     # output folder

# Data
IMAGE_SIZE = 224           # input size
BATCH_SIZE = 32            # batch size
NUM_WORKERS = 2            # dataloader workers
VAL_FRACTION = 0.15        # validation split
TEST_FRACTION = 0.15       # test split
SEED = 42                  # random seed
USE_PATIENT_SPLIT = True   # group by patient_id if available

# Stage 1 (Siamese)
EPOCHS_SIAMESE = 15        # siamese epochs
MARGIN = 0.5               # triplet loss margin
MARGIN_START = 0.3         # starting margin
MARGIN_END = 0.8           # final margin
AUX_LAMBDA = 0.5           # aux loss weight
SUPCON_WARMUP_EPOCHS = 3   # warm-up epochs
SUPCON_TAU = 0.07          # contrastive temp
EMA_DECAY = 0.995          # ema decay
FREEZE_BACKBONE_EPOCHS = 2  # freeze backbone early

# Stage 2 (Classifier)
EPOCHS_CLASSIFIER = 10     # classifier epochs
LR_CLASSIFIER = 1e-3       # classifier lr

# Optimizer
LR_SIAMESE = 1e-4          # siamese lr
WEIGHT_DECAY = 1e-4        # weight decay
BETAS = (0.9, 0.999)       # adam betas
MAX_NORM = 1.0             # grad clip

# Logging
SAVE_PLOTS = True           # save plots
SAVE_MODELS = True          # save models
SMOOTH_K = 1                # smoothing factor
