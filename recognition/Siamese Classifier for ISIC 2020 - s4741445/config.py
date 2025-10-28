DRIVE_ROOT = "/content/drive/MyDrive/siamese_project/data"
IMAGES_DIR = f"{DRIVE_ROOT}/train-image/image"
META_CSV = f"{DRIVE_ROOT}/train-metadata.csv"
ARTIFACTS = "/content/drive/MyDrive/siamese_project/outputs"

IMAGE_SIZE = 224
BATCH_SIZE = 128
NUM_WORKERS = 0
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SEED = 42

# Stage 1 (Siamese) multi-task
EPOCHS_SIAMESE = 12
MARGIN = 0.5
MARGIN_START = 0.3   # schedule to MARGIN
MARGIN_END = 0.8
AUX_LAMBDA = 0.5   # weight on aux CE (anchor)
SUPCON_WARMUP_EPOCHS = 3
SUPCON_TAU = 0.07
EMA_DECAY = 0.995
FREEZE_BACKBONE_EPOCHS = 2
PRETRAINED_BACKBONE = True


# Stage 2 (classifier on cached embeddings)
EPOCHS_CLASSIFIER = 12
LR_CLASSIFIER = 1e-3

# Optim (stage 1)
LR_SIAMESE = 1e-4
WEIGHT_DECAY = 1e-4
BETAS = (0.9, 0.999)
MAX_NORM = 1.0   # grad clip

# Save/plots
SAVE_PLOTS  = True
SAVE_MODELS = True

# Split hygiene
USE_PATIENT_SPLIT = True

# Fast-start / debugging knobs
FAST_DEBUG = True                 # False for full run
FAST_LIMIT_PER_CLASS = 1200       # cap per class if FAST_DEBUG

TRIPLET_STEPS_TRAIN = 80          # steps/epoch Stage-1 (train)
TRIPLET_STEPS_VAL   = 20          # steps/epoch Stage-1 (val)
ASSUME_JPG = True                 # quick path resolve

# Plot smoothing
SMOOTH_K = 3                      # moving-average window for training_plots
