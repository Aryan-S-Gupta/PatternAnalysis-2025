# Paths
DRIVE_ROOT = "/content/drive/MyDrive/siamese_project/data"
IMAGES_DIR = "/content/train-image/image"
META_CSV = "/content/train-metadata.csv"
ARTIFACTS = "/content/drive/MyDrive/siamese_project/outputs"

# Data
IMAGE_SIZE = 224
BATCH_SIZE = 32           # ↓ from 128 for smoother steps (less VRAM stalls)
NUM_WORKERS = 2            # try 4 (or 2 on slow disks); 0 if Windows issues
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SEED = 42
USE_PATIENT_SPLIT = True    # grouped by patient_id; else stratified by target

# Stage 1 (Siamese) multi-task
EPOCHS_SIAMESE = 15   # keep; raise to 16–20 if stable
MARGIN = 0.5
MARGIN_START = 0.3
MARGIN_END = 0.8
AUX_LAMBDA = 0.5
SUPCON_WARMUP_EPOCHS = 3
SUPCON_TAU = 0.07
EMA_DECAY = 0.995
FREEZE_BACKBONE_EPOCHS = 2
PRETRAINED_BACKBONE = True  # always use pretrained CNN backbone

# Stage 2 (classifier on cached embeddings)
EPOCHS_CLASSIFIER = 10          # slight trim for quicker finish
LR_CLASSIFIER = 1e-3

# Optim (stage 1)
LR_SIAMESE = 1e-4
WEIGHT_DECAY = 1e-4
BETAS = (0.9, 0.999)
MAX_NORM = 1.0

# Logging/plots
SAVE_PLOTS = True
SAVE_MODELS = True
SMOOTH_K = 1              # no smoothing → you see true wiggles sooner

# Triplet loader steps (per epoch)
# keep; lower (e.g., 60) if you want even snappier epochs
TRIPLET_STEPS_TRAIN = 80
TRIPLET_STEPS_VAL = 20
