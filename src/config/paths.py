from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent
RESULT_ROOT = SRC_ROOT / 'results'
CKPT_ROOT = SRC_ROOT / 'checkpoints'
CONFIG_ROOT = SRC_ROOT / 'config'
DATA_ROOT = SRC_ROOT.parent / 'tmp'

if not RESULT_ROOT.exists():
    RESULT_ROOT.mkdir()
if not CKPT_ROOT.exists():
    CKPT_ROOT.mkdir()
if not DATA_ROOT.exists():
    DATA_ROOT.mkdir()

VAE_ROOT = SRC_ROOT / 'vae'

DEFAULT_CKPT_FILENAME = 'model.ckpt'
DEFAULT_DATA_DIR = DATA_ROOT / 'data'
if not DEFAULT_DATA_DIR.exists():
    DEFAULT_DATA_DIR.mkdir()
DEFAULT_LOG_DIR = DATA_ROOT / 'logs'
if not DEFAULT_LOG_DIR.exists():
    DEFAULT_LOG_DIR.mkdir()


def build_checkpoint_path(model, latent_dim, hidden_size, layers, i=None):
    path = CKPT_ROOT / model
    if not path.exists():
        path.mkdir()
    path = path / ('%d_%d_%d' % (latent_dim, hidden_size, layers))
    if not path.exists():
        path.mkdir()
    if i is None:
        return '%s/%s' % (str(path), DEFAULT_CKPT_FILENAME)
    else:
        return '%s/%s-%s' % (str(path), DEFAULT_CKPT_FILENAME, i)


def build_results_path(model, latent_dim, hidden_size, layers, i=None):
    path = RESULT_ROOT / model
    if not path.exists():
        path.mkdir()
    path = path / ('%d_%d_%d' % (latent_dim, hidden_size, layers))
    if not path.exists():
        path.mkdir()
    if i is None:
        path = path / 'ckpt'
    else:
        path = path / ('ckpt-%d' % i)
    if not path.exists():
        path.mkdir()
    return str(path)


def build_classifier_checkpoint_path(model, latent_dim, hidden_size, layers, i=None):
    path = CKPT_ROOT / 'classifier'
    if not path.exists():
        path.mkdir()
    path = path / model
    if not path.exists():
        path.mkdir()
    path = path / ('%d_%d_%d' % (latent_dim, hidden_size, layers))
    if not path.exists():
        path.mkdir()
    if i is None:
        return '%s/%s' % (str(path), 'model.h5')
    else:
        return '%s/%s-%s.h5' % (str(path), 'model', i)