<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 12:42:46 2024

@author: Carlito Balingbing
"""

# Import libraries
import os
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow.keras import layers, models
from IPython import display
import pandas as pd
import time

# Static configurations
DATASET_PATH = 'C:/Users/CBalingbing/Insects'
num_threads = 2  # More threads, faster epochs
epoch_count = 50
seed = 42  # Set the seed value for experiment reproducibility
label_names = []


# Define CM save as PNG
def save_plot_as_png(fig, filename):
    directory = 'C:/Users/CBalingbing/CM/Insects'
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    
# Function to set configurations
def set_configs():
    tf.config.threading.set_inter_op_parallelism_threads(num_threads)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Function to squeeze audio
def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels

# Function to get spectrogram
def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

# Function to create spectrogram datasets
def make_spec_ds(ds):
    return ds.map(
        map_func=lambda audio, label: (get_spectrogram(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE)

# Function to print training history information
def print_history_info(history):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    print("Training Loss:", train_loss)
    print("Validation Loss:", val_loss)
    print("Training Accuracy:", train_accuracy)
    print("Validation Accuracy:", val_accuracy)

# Function to load the dataset
def load_dataset(data_dir):
    if not data_dir.exists():
        tf.keras.utils.get_file(
            'insect_sounds',
            origin=DATASET_PATH,
            extract=True,
            cache_dir='.', cache_subdir='data')

# Function to show available directories
def show_available_dirs():
    insects = np.array(tf.io.gfile.listdir(str(data_dir)))
    insects = insects[(insects != 'README.md') & (insects != '.DS_Store')]
    print('Insects:', insects)

# Function to load the training and validation dataset
def load_dataset_train_val():
    tf_data = tf.keras.utils.audio_dataset_from_directory(
        directory=data_dir,
        batch_size=100,
        validation_split=0.2,
        seed=0,
        output_sequence_length=16000,
        subset='both')
    return tf_data

# Load the dataset
data_dir = pathlib.Path(DATASET_PATH)
load_dataset(data_dir)

# List the available folders in the data directory (optional)
show_available_dirs()

# Load audio data for training and validation
train_ds, val_ds = load_dataset_train_val()
label_names = np.array(train_ds.class_names)
num_labels = len(label_names)  # Define num_labels based on the length of label_names
print("\nLabel names:", label_names)

# The dataset now contains batches of audio clips and integer labels.
train_ds.element_spec

val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

# Define the learning rate (choose a suitable value)
learning_rate = 0.0001

# Create spectrogram datasets
train_spectrogram_ds = make_spec_ds(train_ds.map(squeeze, tf.data.AUTOTUNE))
val_spectrogram_ds = make_spec_ds(val_ds.shard(num_shards=2, index=1))
test_spectrogram_ds = make_spec_ds(val_ds.shard(num_shards=2, index=0))

# Examine the spectrograms for different examples of the dataset
for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
    break
    # Create spectrogram datasets
train_spectrogram_ds = make_spec_ds(train_ds.map(squeeze, tf.data.AUTOTUNE))
val_spectrogram_ds = make_spec_ds(val_ds.shard(num_shards=2, index=1))
test_spectrogram_ds = make_spec_ds(val_ds.shard(num_shards=2, index=0))


# Get input_shape from the example spectrograms
for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
    input_shape = example_spectrograms.shape[1:]
    break

input_shape = example_spectrograms.shape[1:]
print('Input shape:', input_shape)
num_labels = len(label_names)

# Instantiate the `tf.keras.layers.Normalization` layer.
norm_layer = layers.Normalization()
# Fit the state of the layer to the spectrograms with `Normalization.adapt`.
norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))

# Function to define the CNN model with regularization
def define_model(input_shape, num_labels, learning_rate, dropout_rate_conv, dropout_rate_dense, l2_reg):
    model = models.Sequential([
       layers.Input(shape=input_shape),
        # Downsample the input.
        layers.Resizing(32, 32),
        # Normalize.
        norm_layer,
        layers.Conv2D(32, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        layers.Dropout(dropout_rate_conv),
        layers.Conv2D(64, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        layers.MaxPooling2D(),
        layers.Dropout(dropout_rate_conv),
        layers.Conv2D(128, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        layers.Dropout(dropout_rate_dense),
        layers.Dense(num_labels),
    ])
    model.summary()

    # Compile the model with the specified learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    return model

# Start the timer
start_time = time.time()

# Adjust hyperparameters for regularization
dropout_rate_conv = 0.25  # Adjust as needed
dropout_rate_dense = 0.5  # Adjust as needed
l2_reg = 0.001  # Adjust as needed
learning_rate =0.0001 # Adjust as needed

# Build a Convolutional Neural Network (CNN) with regularization
model = define_model(input_shape, num_labels, learning_rate, dropout_rate_conv, dropout_rate_dense, l2_reg)

# Train the model over a certain number of epochs
history = model.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=epoch_count,
    #callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)
print('Input shape:', input_shape)

# Access the training and validation metrics from the history object
print_history_info(history)

# Calculate the elapsed time
elapsed_time = time.time() - start_time

# Print the elapsed time in seconds
print("Time taken to train the CNN model: {:.2f} seconds".format(elapsed_time))

# Run the model on the test set and check the model's performance
model.evaluate(test_spectrogram_ds, return_dict=True, verbose=True)

def create_confusion_mtx(test_spectrogram_ds):
    y_pred = model.predict(test_spectrogram_ds)
    y_pred = tf.argmax(y_pred, axis=1)
    y_true = tf.concat(list(test_spectrogram_ds.map(lambda s, lab: lab)), axis=0)

    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)

    # Calculate percentage values for rows and columns separately
    row_sums = tf.reduce_sum(confusion_mtx, axis=1, keepdims=True)
    col_sums = tf.reduce_sum(confusion_mtx, axis=0, keepdims=True)

    percentage_mtx_rows = (confusion_mtx / row_sums) * 100
    percentage_mtx_cols = (confusion_mtx / col_sums) * 100

    # Convert the percentage_mtx_cols to a DataFrame to customize the format
    percentage_df = pd.DataFrame(percentage_mtx_cols.numpy(), columns=label_names, index=label_names)

    fig = plt.figure(figsize=(10, 8))
    # Create the heatmap using percentage_df data and format as percentages
    sns.heatmap(percentage_df, annot=True, fmt='.2f', cmap="gist_earth_r", annot_kws={"size": 14})

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()
   # plt.savefig(OUTPUT_CONFMAT_NAME)

    save_plot_as_png(fig, 'CMInsectsR3.png')

# Use a confusion matrix to check how well the model did classifying each of the commands in the test set
create_confusion_mtx(test_spectrogram_ds)


# Save model and architecture to a single file
save_model_dir = 'C:/Users/CBalingbing/MODELS/CNN/'
model.save(save_model_dir + "CNNInsectsR3_SavedModel.keras") # Save model with .keras extension
print("Saved model to disk", save_model_dir)
=======
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 12:42:46 2024

@author: Carlito Balingbing
"""

# Import libraries
import os
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow.keras import layers, models
from IPython import display
import pandas as pd
import time

# Static configurations
DATASET_PATH = 'C:/Users/CBalingbing/Insects'
num_threads = 2  # More threads, faster epochs
epoch_count = 50
seed = 42  # Set the seed value for experiment reproducibility
label_names = []


# Define CM save as PNG
def save_plot_as_png(fig, filename):
    directory = 'C:/Users/CBalingbing/CM/Insects'
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    
# Function to set configurations
def set_configs():
    tf.config.threading.set_inter_op_parallelism_threads(num_threads)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Function to squeeze audio
def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels

# Function to get spectrogram
def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

# Function to create spectrogram datasets
def make_spec_ds(ds):
    return ds.map(
        map_func=lambda audio, label: (get_spectrogram(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE)

# Function to print training history information
def print_history_info(history):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    print("Training Loss:", train_loss)
    print("Validation Loss:", val_loss)
    print("Training Accuracy:", train_accuracy)
    print("Validation Accuracy:", val_accuracy)

# Function to load the dataset
def load_dataset(data_dir):
    if not data_dir.exists():
        tf.keras.utils.get_file(
            'insect_sounds',
            origin=DATASET_PATH,
            extract=True,
            cache_dir='.', cache_subdir='data')

# Function to show available directories
def show_available_dirs():
    insects = np.array(tf.io.gfile.listdir(str(data_dir)))
    insects = insects[(insects != 'README.md') & (insects != '.DS_Store')]
    print('Insects:', insects)

# Function to load the training and validation dataset
def load_dataset_train_val():
    tf_data = tf.keras.utils.audio_dataset_from_directory(
        directory=data_dir,
        batch_size=100,
        validation_split=0.2,
        seed=0,
        output_sequence_length=16000,
        subset='both')
    return tf_data

# Load the dataset
data_dir = pathlib.Path(DATASET_PATH)
load_dataset(data_dir)

# List the available folders in the data directory (optional)
show_available_dirs()

# Load audio data for training and validation
train_ds, val_ds = load_dataset_train_val()
label_names = np.array(train_ds.class_names)
num_labels = len(label_names)  # Define num_labels based on the length of label_names
print("\nLabel names:", label_names)

# The dataset now contains batches of audio clips and integer labels.
train_ds.element_spec

val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

# Define the learning rate (choose a suitable value)
learning_rate = 0.0001

# Create spectrogram datasets
train_spectrogram_ds = make_spec_ds(train_ds.map(squeeze, tf.data.AUTOTUNE))
val_spectrogram_ds = make_spec_ds(val_ds.shard(num_shards=2, index=1))
test_spectrogram_ds = make_spec_ds(val_ds.shard(num_shards=2, index=0))

# Examine the spectrograms for different examples of the dataset
for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
    break
    # Create spectrogram datasets
train_spectrogram_ds = make_spec_ds(train_ds.map(squeeze, tf.data.AUTOTUNE))
val_spectrogram_ds = make_spec_ds(val_ds.shard(num_shards=2, index=1))
test_spectrogram_ds = make_spec_ds(val_ds.shard(num_shards=2, index=0))


# Get input_shape from the example spectrograms
for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
    input_shape = example_spectrograms.shape[1:]
    break

input_shape = example_spectrograms.shape[1:]
print('Input shape:', input_shape)
num_labels = len(label_names)

# Instantiate the `tf.keras.layers.Normalization` layer.
norm_layer = layers.Normalization()
# Fit the state of the layer to the spectrograms with `Normalization.adapt`.
norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))

# Function to define the CNN model with regularization
def define_model(input_shape, num_labels, learning_rate, dropout_rate_conv, dropout_rate_dense, l2_reg):
    model = models.Sequential([
       layers.Input(shape=input_shape),
        # Downsample the input.
        layers.Resizing(32, 32),
        # Normalize.
        norm_layer,
        layers.Conv2D(32, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        layers.Dropout(dropout_rate_conv),
        layers.Conv2D(64, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        layers.MaxPooling2D(),
        layers.Dropout(dropout_rate_conv),
        layers.Conv2D(128, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        layers.Dropout(dropout_rate_dense),
        layers.Dense(num_labels),
    ])
    model.summary()

    # Compile the model with the specified learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    return model

# Start the timer
start_time = time.time()

# Adjust hyperparameters for regularization
dropout_rate_conv = 0.25  # Adjust as needed
dropout_rate_dense = 0.5  # Adjust as needed
l2_reg = 0.001  # Adjust as needed
learning_rate =0.0001 # Adjust as needed

# Build a Convolutional Neural Network (CNN) with regularization
model = define_model(input_shape, num_labels, learning_rate, dropout_rate_conv, dropout_rate_dense, l2_reg)

# Train the model over a certain number of epochs
history = model.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=epoch_count,
    #callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)
print('Input shape:', input_shape)

# Access the training and validation metrics from the history object
print_history_info(history)

# Calculate the elapsed time
elapsed_time = time.time() - start_time

# Print the elapsed time in seconds
print("Time taken to train the CNN model: {:.2f} seconds".format(elapsed_time))

# Run the model on the test set and check the model's performance
model.evaluate(test_spectrogram_ds, return_dict=True, verbose=True)

def create_confusion_mtx(test_spectrogram_ds):
    y_pred = model.predict(test_spectrogram_ds)
    y_pred = tf.argmax(y_pred, axis=1)
    y_true = tf.concat(list(test_spectrogram_ds.map(lambda s, lab: lab)), axis=0)

    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)

    # Calculate percentage values for rows and columns separately
    row_sums = tf.reduce_sum(confusion_mtx, axis=1, keepdims=True)
    col_sums = tf.reduce_sum(confusion_mtx, axis=0, keepdims=True)

    percentage_mtx_rows = (confusion_mtx / row_sums) * 100
    percentage_mtx_cols = (confusion_mtx / col_sums) * 100

    # Convert the percentage_mtx_cols to a DataFrame to customize the format
    percentage_df = pd.DataFrame(percentage_mtx_cols.numpy(), columns=label_names, index=label_names)

    fig = plt.figure(figsize=(10, 8))
    # Create the heatmap using percentage_df data and format as percentages
    sns.heatmap(percentage_df, annot=True, fmt='.2f', cmap="gist_earth_r", annot_kws={"size": 14})

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()
   # plt.savefig(OUTPUT_CONFMAT_NAME)

    save_plot_as_png(fig, 'CMInsectsR3.png')

# Use a confusion matrix to check how well the model did classifying each of the commands in the test set
create_confusion_mtx(test_spectrogram_ds)


# Save model and architecture to a single file
save_model_dir = 'C:/Users/CBalingbing/MODELS/CNN/'
model.save(save_model_dir + "CNNInsectsR3_SavedModel.keras") # Save model with .keras extension
print("Saved model to disk", save_model_dir)
>>>>>>> 260e25da551e25a749a0f09b67031d2f2653fd19
