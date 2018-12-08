from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent

RESULT_DIR = SRC_ROOT / 'results'
if not RESULT_DIR.exists():
    RESULT_DIR.mkdir()

CHECKPOINT_DIR = SRC_ROOT / 'checkpoints'
if not CHECKPOINT_DIR.exists():
    CHECKPOINT_DIR.mkdir()

MODEL_OUTPUT_FILENAME = 'model.ckpt'
DEFAULT_EPOCHS = 75

DEFAULT_VAE_CONFIG = 'mnist_vae'


def build_checkpoint_path(config, filename):
    path = CHECKPOINT_DIR / config / filename
    if not path.exists():
        path.mkdir()
    return str(path)