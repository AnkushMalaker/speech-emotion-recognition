from sklearn.utils import validation
from tensorflow.python.keras.engine import training
from utils import get_dataset, create_model
import tensorflow as tf
import random
import argparse
import warnings

# This is required to avoid librosa's warning about n_fft being too large. I don't know how to fix the issue the warning is trying to fix since
# Specifying a smaller n_ftt doesn't seem to fix it.
warnings.filterwarnings("ignore", category=UserWarning)

# Argument Parser

help_message = "Check the documentation available at https://www.github.com/AnkushMalaker/speech-emotion-recognition for more info on usage."
parser = argparse.ArgumentParser(description=help_message)

parser.add_argument("epochs", type=int, help="Specify number of epochs")

parser.add_argument(
    "-B",
    "--batch_size",
    type=int,
    help="Default batch size is 32. Reduce this if the data doesn't fit in your GPU.",
)

parser.add_argument(
    "-C",
    "--cache",
    action="store_true",
    help="Default behaviour is to not use cahce. Caching greatly speeds up the training after 1 epoch but may require a lot of Memory.",
)

parser.add_argument(
    "-LR", "--learning_rate", type=float, help="Default Learning rate is 1e-5."
)

parser.add_argument("--train_dir", help="Default data directory is ./train_data")

parser.add_argument(
    "--val_dir",
    help="Default behaviour is to take given split from train_dir to do validation. Specify the split using --val_split",
)

parser.add_argument(
    "--val_split", type=float, help="Default val_split is 0.2 of train data"
)

parser.add_argument(
    "--dataset",
    help='Specifies the specific architecture to be used. Check README for more info. Defaults to "emoDB".',
)

# This logic has to be improved to adapt to other datasets mentioned in the paper.
# Or the whole code has to be split into different sections for each, like train_emodb.py, train_xyz,... etc.
# Currently not handling this argument
parser.add_argument("--num_labels", help="Specify number of labels")

parser.add_argument(
    "--random_state",
    type=int,
    help="Specify random state for consistency in experiments. Use -1 to randomize.",
)

args = parser.parse_args()

EPOCHS = args.epochs

print(EPOCHS)
if args.cache:
    CACHE = True
else:
    CACHE = False

if args.batch_size:
    BATCH_SIZE = args.batch_size
else:
    BATCH_SIZE = 32

if args.train_dir:
    train_dir = args.train_dir
else:
    train_dir = "./train_data"

if args.val_dir:
    val_dir = args.val_dir
else:
    val_dir = None

if args.random_state == -1:
    RANDOM_STATE = random.randint(0, 10000)
else:
    RANDOM_STATE = 42

if args.val_split:
    val_split = args.val_split
else:
    val_split = 0.2

train_ds, val_ds = get_dataset(
    training_dir=train_dir,
    validation_dir=val_dir,
    val_split=val_split,
    batch_size=BATCH_SIZE,
    random_state=RANDOM_STATE,
    cache=CACHE,
)

model = create_model()

ESCallback = tf.keras.callbacks.EarlyStopping(
    patience=2, restore_best_weights=True, verbose=3
)

# Add checkpoint callback
model.fit(
    train_ds,
    validation_data=val_ds,
    callbacks=ESCallback,
    epochs=EPOCHS,
)

model.save(f'saved_model/{EPOCHS}_trained_model')