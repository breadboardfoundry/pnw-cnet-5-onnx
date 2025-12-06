# 13 Jun 2023
# Script for training PNW-Cnet v5.

import glob 
import h5py
import os
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from tensorflow.keras.callbacks import Callback, CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Attempt to avoid CUDA_ERROR_OUT_OF_MEMORY issues - by default TensorFlow is
# very greedy and attempts to allocate all GPU memory; this allows nicer sharing 
# among multiple GPUs
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


training_dir = "./data"
output_dir = "./output"


# Contains paths to all images and their labels in text format. Script
# will generate tags in binary n-hot format and separate the full set
# into training (85%), validation (10%), and test (5%) sets.
train_file_path = os.path.join(training_dir, "Training_Set.csv")

for i in range(10):
    variant = "abcdefghij"[i]
    model_name = "Test_Model_%s%s.h5" % (time.strftime("%d%b%y"), variant)
    h5_model_path = Path(output_dir, model_name)
    if not h5_model_path.exists():
        break

# A few general parameters to control training behavior
batchsize = 256
initial_learning_rate = 0.0015 
n_epochs = 30
n_fc_nodes = 512
dropout_prop = 0.30

# Set to < 1 to use a random subset of the complete training set for 
# smaller tests.
train_set_fraction = 1.0

# Constructs the model and returns it for compilation and fitting
def build_model(nclasses): #, optimizer):
    model = Sequential()

    model.add(Conv2D(32, (5,5), 
                input_shape = (257,1000,1), 
                data_format='channels_last', 
                activation = 'relu', 
                padding = 'same'))

    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(dropout_prop))

    model.add(Conv2D(32, (5,5), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(dropout_prop))

    model.add(Conv2D(64, (5,5), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(dropout_prop))

    model.add(Conv2D(64, (5,5), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(dropout_prop))

    model.add(Conv2D(128, (5,5), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(dropout_prop))

    model.add(Conv2D(128, (5,5), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(dropout_prop))

    model.add(Flatten())

    model.add(Dense(n_fc_nodes, activation = 'relu'))
    model.add(Dropout(dropout_prop))

    model.add(Dense(nclasses, activation = 'sigmoid'))

    return model


# Retrieves current learning rate from optimizer at each epoch
def get_learning_rate(optimizer):
    lr = optimizer.learning_rate
    if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
        return lr(optimizer.iterations)
    else:
        return lr


# Like CSVLogger but also writes the time when each epoch ended, the learning
# rate, and whether or not the model was saved.
class ModelLogger(Callback):
    def __init__(self, model_path):
        self.model_path = model_path
        self.log_path = model_path.replace(".h5", "_training_log.csv")

    def on_train_begin(self, logs=None):
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss")
        if val_loss < self.best:
            model_saved = "Y"
            self.best = val_loss
        else:
            model_saved = "N"
        # lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        lr = get_learning_rate(self.model.optimizer)
        str_time = time.strftime("%m/%d/%y %H:%M:%S")
        log_line = "Epoch_{0:02d},{1},{2:.4f},{3:.4f},{4:.4f},{5:.4f},{6:.6f},{7}\n".format(epoch+1, str_time, logs["accuracy"], logs["loss"], logs["val_accuracy"], logs["val_loss"], lr, model_saved)
        with open(self.log_path, 'a') as log_file:
            if epoch == 0:
                log_file.write("Epoch,Time,Accuracy,Loss,Val_Accuracy,Val_Loss,Model_Saved\n")
            log_file.write(log_line)


def main():
    starttime = time.time()
    print("Starting at {0}...".format(time.strftime("%H:%M:%S")))

    class_list = "ACCO1,ACGE1,ACGE2,ACST1,AEAC1,AEAC2,Airplane,ANCA1,ASOT1,BOUM1,BRCA1,BRMA1,BRMA2,BUJA1,BUJA2,Bullfrog,BUVI1,BUVI2,CACA1,CAGU1,CAGU2,CAGU3,CALA1,CALU1,CAPU1,CAUS1,CAUS2,CCOO1,CCOO2,CECA1,Chainsaw,CHFA1,Chicken,CHMI1,CHMI2,COAU1,COAU2,COBR1,COCO1,COSO1,Cow,Creek,Cricket,CYST1,CYST2,DEFU1,DEFU2,Dog,DRPU1,Drum,EMDI1,EMOB1,FACO1,FASP1,Fly,Frog,GADE1,GLGN1,Growler,Gunshot,HALE1,HAPU1,HEVE1,Highway,Horn,Human,HYPI1,IXNA1,IXNA2,JUHY1,LEAL1,LECE1,LEVI1,LEVI2,LOCU1,MEFO1,MEGA1,MEKE1,MEKE2,MEKE3,MYTO1,NUCO1,OCPR1,ODOC1,ORPI1,ORPI2,PAFA1,PAFA2,PAHA1,PECA1,PHME1,PHNU1,PILU1,PILU2,PIMA1,PIMA2,POEC1,POEC2,PSFL1,Rain,Raptor,SICU1,SITT1,SITT2,SPHY1,SPHY2,SPPA1,SPPI1,SPTH1,STDE1,STNE1,STNE2,STOC_4Note,STOC_Series,Strix_Bark,Strix_Whistle,STVA_8Note,STVA_Insp,STVA_Series,Survey_Tone,TADO1,TADO2,TAMI1,Thunder,TRAE1,Train,Tree,TUMI1,TUMI2,URAM1,VIHU1,Wildcat,Yarder,ZEMA1,ZOLE1".split(',')

    n_classes = len(class_list) # == 135 with above list

    # Define train-validation-test split
    split_csv = Path(training_dir, "Train_Val_Test_Split.csv")
    test_csv = Path(training_dir, "Test_Set.csv")

    if not os.path.exists(split_csv):
        # Reading in our training data info as a pandas data frame.
        train_df = pd.read_csv(train_file_path)
        
        labels_nhot = [[1 if i in tags.split('+') else 0 for tags in df["Tags"]] for i in class_list]
        label_df = pd.DataFrame(dict(zip(class_list, labels_nhot)))
        
        train_df = pd.concat([train_df, label_df], axis=1)

        train_df["Path"] = train_df.apply(lambda row: Path(training_dir, "images", row.Filename), axis = 1)
        print("Found {0} images with {1} unique classes.".format(len(train_df), n_classes))

        # Split the data into training and validation sets, both dataframes.
        # Select a random subset of rows for the validation set, then drop
        # those rows from the original dataframe.
        val_set = train_df.sample(frac = 0.1, random_state = 651)
        train_set = train_df.drop(val_set.index)
        
        # Put the result of the training-validation split into a single table and 
        # write it to a csv file so we know which images were used in each set.
        train_set.insert(1, "Set", ["Training" for i in range(len(train_set))])
        val_set.insert(1, "Set", ["Validation" for i in range(len(val_set))])
        
        # Do the same for the test set (5% of the full set)
        test_set = train_set.sample(frac = 0.05555, random_state = 651)
        train_set = train_set.drop(test_set.index)
        test_set["Set"] = ["Test" for i in range(len(test_set))]
        
        out_table = train_set.append(val_set).append(test_set)
        out_table.sort_values(by=["Set", "Filename"], inplace=True)
        
        out_table.to_csv(split_csv, index=False)
        test_set.to_csv(test_csv, index=False)

    else:
        split_df = pd.read_csv(split_csv)
        train_set = split_df[split_df["Set"] == "Training"]
        val_set = split_df[split_df["Set"] == "Validation"]

    if train_set_fraction < 1.0:
        train_set = train_set.sample(frac = train_set_fraction)
        val_set = val_set.sample(frac = train_set_fraction)

    n_train, n_val = len(train_set), len(val_set)

    # Constructing and compiling the model
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = initial_learning_rate,
        decay_steps = 10000,
        decay_rate = 0.96)

    adam = Adam(learning_rate = learning_rate_schedule)

    # Attempt to set up a "mirrored strategy" to duplicate the model and split 
    # batches across all available GPUs. Doesn't seem to work with SGE_Batch; use 
    # screen to run in interactive mode and keep it going after session ends.
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = build_model(n_classes)
        model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics=['accuracy'])
    print("\n\nModel compiled successfully.\n")
    print("Using {0:.1f}% of full training set ({1} images) for test purposes.\n".format(train_set_fraction * 100, n_train + n_val))

    # Callback functions to dictate training behavior. This says we save the
    # model only when validation loss decreases (prevents overfitting).
    best_fit = ModelCheckpoint(
        str(h5_model_path), 
        monitor = 'val_loss', 
        verbose = 1,
        save_best_only = True)

    model_logger = ModelLogger(h5_model_path)

    # Rescales the pixel values in each image to floats in the range [0, 1].
    image_gen = ImageDataGenerator(rescale=1./255)

    # Reads filenames and labels from the dataframe and supplies them to the
    # model for training.
    train_data = image_gen.flow_from_dataframe(
        dataframe = train_set, 
        directory = None, 
        x_col = "Path",
        y_col = class_list,
        target_size = (257,1000),
        batch_size = batchsize,
        color_mode = "grayscale",
        class_mode = "raw")

    # Same thing but for the validation dataset.
    val_data = image_gen.flow_from_dataframe(
        dataframe = val_set, 
        directory = None,
        x_col = "Path",
        y_col = class_list,
        target_size = (257,1000),
        batch_size = batchsize,
        color_mode = "grayscale",
        class_mode = "raw")

    # Trains the model using the training and validation sets previously generated.
    model_hist = model.fit(
        x = train_data,
        steps_per_epoch = int(n_train / batchsize) + 1,
        epochs = n_epochs,
        verbose = 1,
        validation_data = val_data,
        validation_steps = int(n_val / batchsize) + 1,
        callbacks = [best_fit, model_logger])

    elapsed = time.time() - starttime

    print("Training run complete. {0:.1f} seconds elapsed.".format(elapsed))

if __name__ == "__main__":
    main()
