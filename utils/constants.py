import os
import enum


BINARIES_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'binaries')
CHECKPOINTS_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'checkpoints')
DATA_DIR_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'data')

# Make sure these exist as the rest of the code assumes it
os.makedirs(BINARIES_PATH, exist_ok=True)
os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
os.makedirs(DATA_DIR_PATH, exist_ok=True)

LATENT_SPACE_DIM = 100  # input random vector size to generator network
MNIST_IMG_SIZE = 28
MNIST_NUM_CLASSES = 10


class GANType(enum.Enum):
    VANILLA = 0,
    CGAN = 1,
    DCGAN = 2

