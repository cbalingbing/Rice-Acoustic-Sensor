# -*- coding: utf-8 -*-
"""
Created on Sat 6 June 09:25:11 2024

@author: Carlito Balingbing
"""

# import libraries
import os
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow.keras import layers
from tensorflow.keras import models
import pandas as pd
from sklearn import metrics
import mplcyberpunk
import time


# static configurations
DATASET_PATH = 'C:/Users/CBalingbing/Insects'
#OUTPUT_MODEL_NAME = 'CNNmodeCNN1.keras'
#OUTPUT_CONFMAT_NAME = 'confusion_mtx_insects.png'
num_threads = 2 # more threads, faster epochs
epoch_count = 30
seed = 42 # Set the seed value for experiment reproducibility.
label_names = []

#Define CM save as PNG
def save_plot_as_png(fig, filename):
    directory = 'C:/Users/CBalingbing/CM/Insects'
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')

######################################
def set_configs(): 
  tf.config.threading.set_inter_op_parallelism_threads(num_threads)
  np.random.seed(seed)
  tf.random.set_seed(seed)

# This dataset only contains single channel audio, so use the `tf.squeeze` function to drop the extra axis:
def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels
  
# Function to create spectrogram datasets
def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def make_spec_ds(ds):
    return ds.map(
        map_func=lambda audio, label: (get_spectrogram(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE)

#Load the data set
def load_dataset(data_dir):
  if not data_dir.exists():
    tf.keras.utils.get_file(
        'insect_sounds',
        origin=DATASET_PATH,
        extract=True,
        cache_dir='.', cache_subdir='data')

def show_available_dirs():
  insects = np.array(tf.io.gfile.listdir(str(data_dir)))
  insects = insects[(insects != 'README.md') & (insects != '.DS_Store')]
  print('Insects:', insects)

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
# Fit the state of the layer to the spectrograms
# with `Normalization.adapt`.
norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))



# Build a Convolutional Neural Network (CNN)
model = models.Sequential([
    layers.Input(shape=input_shape),
    # Downsample the input.
    layers.Resizing(32, 32),
    # Normalize.
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels),
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

# Train the model
history = model.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=epoch_count,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
  )

# Start the timer
start_time = time.time()

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

train_loss=history.history['loss']
val_loss=history.history['val_loss']
train_accuracy=history.history['accuracy']
val_accuracy=history.history['val_accuracy']

# Set plot style
plt.style.use("cyberpunk")


# Create a figure
plt.figure(figsize=(12, 6))

# Plot training & validation loss values
plt.plot(train_loss, label='Training Loss', color='cyan')
plt.plot(val_loss, label='Validation Loss', color='cyan', linestyle='dashed')

# Plot training & validation accuracy values
plt.plot(train_accuracy, label='Training Accuracy', color='magenta')
plt.plot(val_accuracy, label='Validation Accuracy', color='magenta', linestyle='dashed')

# Set titles and labels
plt.title('Loss and Accuracy of a CNN model for insect sound classification using Spectrogram features')
plt.xlabel('Epoch')
plt.ylabel('Value')

# Add a legend
plt.legend(loc='best')
# Save the plot as PNG
plt.savefig('C:/Users/CBalingbing/PLOTS/Insects/ValLoss_spectrograms.png')
# Access the training and validation metrics from the history object
#print_history_info(history)
plt.show()

# Calculate the elapsed time
elapsed_time = time.time() - start_time

# Print the elapsed time in seconds
print("Time taken to train the CNN model: {:.2f} seconds".format(elapsed_time))

# Evaluate the model using the test MFCC dataset
evaluation = model.evaluate(test_spectrogram_ds, return_dict=True)

# Predict on the test dataset
y_pred = model.predict(test_spectrogram_ds)
y_pred = tf.argmax(y_pred, axis=1)
y_true = tf.concat(list(test_spectrogram_ds.map(lambda s, lab: lab)), axis=0)

# Create confusion matrix
confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)

# Calculate percentage values for rows and columns separately
row_sums = tf.reduce_sum(confusion_mtx, axis=1, keepdims=True)
col_sums = tf.reduce_sum(confusion_mtx, axis=0, keepdims=True)

percentage_mtx_rows = (confusion_mtx / row_sums) * 100
percentage_mtx_cols = (confusion_mtx / col_sums) * 100

# Convert to DataFrame for better visualization
#label_names = ['R_dominica', 'S_oryzae', 'T_castaneum']  # Replace with actual class names
percentage_df = pd.DataFrame(percentage_mtx_cols.numpy(), index=label_names, columns=label_names)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(percentage_df, annot=True, fmt='.2f', cmap="gist_earth_r", annot_kws={"size": 14})
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.title('Confusion Matrix of Insect Sound Classification via CNN using Spectrogram Features')
plt.savefig('C:/Users/CBalingbing/CM/Insects/ConfusionMatrixInsects.png')
plt.show()

# Save the model
save_model_dir = 'C:/Users/CBalingbing/MODELS/CNN/'
model.save(save_model_dir + "CNNInsects_SavedModel.keras")
print("Saved model to disk:", save_model_dir)