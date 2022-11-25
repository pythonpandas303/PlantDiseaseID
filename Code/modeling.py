### Final Project
### MSDS696
### Tom Teasdale

### Image classification for plant pathology data set. Currently the final model is the most 
### accurate and should be applied to all future data sets. Final chunk is wrapping into 
### TF Lite. 


# Importing required libraries

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pathlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import glob
import os
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
import math

# Setting data set path, printing image count

dataset_path = "/content/drive/MyDrive/Plant"
data_dir = pathlib.Path(dataset_path)
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

# Opening random photo from set

complexex = list(data_dir.glob('complex/*'))
PIL.Image.open(str(complexex[15]))

# Defining batch and image size

batch_size = 36
img_height = 128
img_width = 128

# Definition of train and val data sets

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Coutning and printing class names

class_names = train_ds.class_names
print(class_names)

# =============================================================================
# ['complex', 'frog_eye_leaf_spot', 'frog_eye_leaf_spot complex', 'healthy', 
#  'powdery_mildew', 'powdery_mildew complex', 'rust', 'rust complex', 
#  'rust frog_eye_leaf_spot', 'scab', 'scab frog_eye_leaf_spot', 
#  'scab frog_eye_leaf_spot complex']
# =============================================================================

# Plotting of test images

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
    
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

# Autotuning

AUTOTUNE = tf.data.AUTOTUNE

# Prefetching to decrease memory cost

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Definition of normalization layer and normalizing data set

normalization_layer = layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

# Building model in Keras API

num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

# Compilation of model

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Printing model summary

model.summary()

# =============================================================================
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  rescaling_1 (Rescaling)     (None, 128, 128, 3)       0         
#                                                                  
#  conv2d (Conv2D)             (None, 128, 128, 16)      448       
#                                                                  
#  max_pooling2d (MaxPooling2D  (None, 64, 64, 16)       0         
#  )                                                               
#                                                                  
#  conv2d_1 (Conv2D)           (None, 64, 64, 32)        4640      
#                                                                  
#  max_pooling2d_1 (MaxPooling  (None, 32, 32, 32)       0         
#  2D)                                                             
#                                                                  
#  conv2d_2 (Conv2D)           (None, 32, 32, 64)        18496     
#                                                                  
#  max_pooling2d_2 (MaxPooling  (None, 16, 16, 64)       0         
#  2D)                                                             
#                                                                  
#  flatten (Flatten)           (None, 16384)             0         
#                                                                  
#  dense (Dense)               (None, 128)               2097280   
#                                                                  
#  dense_1 (Dense)             (None, 12)                1548      
#                                                                  
# =================================================================
# Total params: 2,122,412
# Trainable params: 2,122,412
# Non-trainable params: 0
# _________________________________________________________________
# 
# =============================================================================

# Model run, adjust epochs, callbacks and others as necessary 

epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# =============================================================================
# Epoch 1/10
# 415/415 [==============================] - 475s 1s/step - loss: 1.6495 - accuracy: 0.3882 - val_loss: 1.4687 - val_accuracy: 0.4748
# Epoch 2/10
# 415/415 [==============================] - 162s 390ms/step - loss: 1.3446 - accuracy: 0.5257 - val_loss: 1.3213 - val_accuracy: 0.5341
# Epoch 3/10
# 415/415 [==============================] - 165s 398ms/step - loss: 1.0997 - accuracy: 0.6183 - val_loss: 1.1515 - val_accuracy: 0.6079
# Epoch 4/10
# 415/415 [==============================] - 162s 391ms/step - loss: 0.8756 - accuracy: 0.6917 - val_loss: 1.0907 - val_accuracy: 0.6318
# Epoch 5/10
# 415/415 [==============================] - 163s 393ms/step - loss: 0.7051 - accuracy: 0.7480 - val_loss: 1.0688 - val_accuracy: 0.6331
# Epoch 6/10
# 415/415 [==============================] - 163s 393ms/step - loss: 0.5041 - accuracy: 0.8221 - val_loss: 1.1605 - val_accuracy: 0.6409
# Epoch 7/10
# 415/415 [==============================] - 163s 393ms/step - loss: 0.3198 - accuracy: 0.8890 - val_loss: 1.3166 - val_accuracy: 0.6645
# Epoch 8/10
# 415/415 [==============================] - 166s 399ms/step - loss: 0.1895 - accuracy: 0.9362 - val_loss: 1.5893 - val_accuracy: 0.6565
# Epoch 9/10
# 415/415 [==============================] - 166s 399ms/step - loss: 0.1116 - accuracy: 0.9662 - val_loss: 1.7341 - val_accuracy: 0.6490
# Epoch 10/10
# 415/415 [==============================] - 166s 400ms/step - loss: 0.0840 - accuracy: 0.9746 - val_loss: 1.9374 - val_accuracy: 0.6492
# =============================================================================

# Plotting of model run

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Definition of data sugmentation

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

# Plotting of test images

plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")
    
# Building model in Keras API
    
model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, name="outputs")
])

# Compilation of model

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Model run, adjust epochs, callbacks and others as necessary 

epochs = 15
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# =============================================================================
# Epoch 1/15
# 415/415 [==============================] - 195s 466ms/step - loss: 1.6905 - accuracy: 0.3762 - val_loss: 1.4723 - val_accuracy: 0.4801
# Epoch 2/15
# 415/415 [==============================] - 191s 462ms/step - loss: 1.3603 - accuracy: 0.5199 - val_loss: 1.2205 - val_accuracy: 0.5733
# Epoch 3/15
# 415/415 [==============================] - 193s 465ms/step - loss: 1.1567 - accuracy: 0.5973 - val_loss: 1.0596 - val_accuracy: 0.6208
# Epoch 4/15
# 415/415 [==============================] - 195s 470ms/step - loss: 1.0163 - accuracy: 0.6440 - val_loss: 0.9716 - val_accuracy: 0.6731
# Epoch 5/15
# 415/415 [==============================] - 197s 476ms/step - loss: 0.9294 - accuracy: 0.6824 - val_loss: 0.9481 - val_accuracy: 0.6822
# Epoch 6/15
# 415/415 [==============================] - 195s 470ms/step - loss: 0.8550 - accuracy: 0.7085 - val_loss: 0.9152 - val_accuracy: 0.6876
# Epoch 7/15
# 415/415 [==============================] - 192s 463ms/step - loss: 0.7845 - accuracy: 0.7315 - val_loss: 0.8577 - val_accuracy: 0.6852
# Epoch 8/15
# 415/415 [==============================] - 192s 462ms/step - loss: 0.7402 - accuracy: 0.7481 - val_loss: 0.8387 - val_accuracy: 0.7292
# Epoch 9/15
# 415/415 [==============================] - 191s 461ms/step - loss: 0.7117 - accuracy: 0.7586 - val_loss: 0.7387 - val_accuracy: 0.7547
# Epoch 10/15
# 415/415 [==============================] - 194s 466ms/step - loss: 0.6757 - accuracy: 0.7722 - val_loss: 0.7482 - val_accuracy: 0.7523
# Epoch 11/15
# 415/415 [==============================] - 193s 464ms/step - loss: 0.6664 - accuracy: 0.7752 - val_loss: 0.7615 - val_accuracy: 0.7560
# Epoch 12/15
# 415/415 [==============================] - 192s 463ms/step - loss: 0.6340 - accuracy: 0.7898 - val_loss: 0.6876 - val_accuracy: 0.7794
# Epoch 13/15
# 415/415 [==============================] - 193s 464ms/step - loss: 0.6213 - accuracy: 0.7919 - val_loss: 0.6792 - val_accuracy: 0.7864
# Epoch 14/15
# 415/415 [==============================] - 192s 462ms/step - loss: 0.6005 - accuracy: 0.7988 - val_loss: 0.6820 - val_accuracy: 0.7880
# Epoch 15/15
# 415/415 [==============================] - 192s 462ms/step - loss: 0.5776 - accuracy: 0.8102 - val_loss: 0.6433 - val_accuracy: 0.7928
# =============================================================================

# Plotting model run

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Test Prediction, Change file path as necessary

prediction_path = "/content/drive/MyDrive/powdery-mildew-lesions.jpg"


img = tf.keras.utils.load_img(
    prediction_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

# =============================================================================
# 1/1 [==============================] - 0s 137ms/step
# This image most likely belongs to powdery_mildew with a 67.93 percent confidence.
# =============================================================================

# Setting dataset path, counting images and printing image count

dataset_path = "/content/drive/MyDrive/Tomato"
data_dir = pathlib.Path(dataset_path)
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

# Definition of batch size and image size

batch_size = 18
img_height = 128
img_width = 128

# Definition of data sets

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Defining and printing class names

class_names = train_ds.class_names
print(class_names)

# =============================================================================
# ['Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 'Septoria_leaf_spot', 
#  'Spider_mites Two-spotted_spider_mite', 'Target_Spot', 'Tomato_Yellow_Leaf_Curl_Virus', 
#  'Tomato_mosaic_virus', 'healthy', 'powdery_mildew']
# =============================================================================

# Plotting of test images

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
    
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

# Autotuning

AUTOTUNE = tf.data.AUTOTUNE

# Prefetching to decrease memory cost

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Calculating normalization layer and definition of normalized data set

normalization_layer = layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

# Plotting of test pictures

plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")

# Definition of Early Stopping Callback

early_stop = EarlyStopping(monitor='val_loss',
                           patience=10,
                           restore_best_weights=True,
                           mode='min')

# Building model in Keras API

num_classes = len(class_names)

model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, name="outputs")
])

# Compilation of model

model.compile(optimizer='adam'(leanning_rate=0.00001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Computation of steps per epoch, adjust training and val size according to code executed above

compute_steps_per_epoch = lambda x: int(math.ceil(1. * x / batch_size))
training_size = 20668
val_size = 5167
steps_per_epoch = compute_steps_per_epoch(training_size)
val_steps = compute_steps_per_epoch(val_size)

# Model run, adjust epochs, callbacks and others as necessary 

epochs = 25
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=early_stop,
  steps_per_epoch=steps_per_epoch
)

# =============================================================================
# Epoch 1/25
# 575/575 [==============================] - 60s 104ms/step - loss: 0.9586 - accuracy: 0.6669 - val_loss: 0.8713 - val_accuracy: 0.7111
# Epoch 2/25
# 575/575 [==============================] - 12s 21ms/step - loss: 0.7842 - accuracy: 0.7317 - val_loss: 0.8141 - val_accuracy: 0.7167
# Epoch 3/25
# 575/575 [==============================] - 12s 21ms/step - loss: 0.6845 - accuracy: 0.7633 - val_loss: 0.8433 - val_accuracy: 0.7087
# Epoch 4/25
# 575/575 [==============================] - 12s 22ms/step - loss: 0.6166 - accuracy: 0.7872 - val_loss: 0.8404 - val_accuracy: 0.7234
# Epoch 5/25
# 575/575 [==============================] - 12s 21ms/step - loss: 0.5754 - accuracy: 0.7955 - val_loss: 0.8412 - val_accuracy: 0.7263
# Epoch 6/25
# 575/575 [==============================] - 12s 21ms/step - loss: 0.5375 - accuracy: 0.8149 - val_loss: 0.5290 - val_accuracy: 0.8192
# Epoch 7/25
# 575/575 [==============================] - 12s 21ms/step - loss: 0.5143 - accuracy: 0.8213 - val_loss: 0.6407 - val_accuracy: 0.7869
# Epoch 8/25
# 575/575 [==============================] - 12s 21ms/step - loss: 0.4882 - accuracy: 0.8332 - val_loss: 0.5368 - val_accuracy: 0.8138
# Epoch 9/25
# 575/575 [==============================] - 12s 21ms/step - loss: 0.4635 - accuracy: 0.8371 - val_loss: 0.6343 - val_accuracy: 0.7908
# Epoch 10/25
# 575/575 [==============================] - 12s 21ms/step - loss: 0.4254 - accuracy: 0.8509 - val_loss: 0.6392 - val_accuracy: 0.7900
# Epoch 11/25
# 575/575 [==============================] - 12s 21ms/step - loss: 0.4250 - accuracy: 0.8522 - val_loss: 0.6598 - val_accuracy: 0.7865
# Epoch 12/25
# 575/575 [==============================] - 12s 21ms/step - loss: 0.4093 - accuracy: 0.8558 - val_loss: 0.5052 - val_accuracy: 0.8401
# Epoch 13/25
# 575/575 [==============================] - 12s 22ms/step - loss: 0.3962 - accuracy: 0.8625 - val_loss: 0.4264 - val_accuracy: 0.8560
# Epoch 14/25
# 575/575 [==============================] - 12s 21ms/step - loss: 0.3859 - accuracy: 0.8662 - val_loss: 0.4796 - val_accuracy: 0.8386
# Epoch 15/25
# 575/575 [==============================] - 12s 21ms/step - loss: 0.3590 - accuracy: 0.8754 - val_loss: 0.4862 - val_accuracy: 0.8417
# Epoch 16/25
# 575/575 [==============================] - 12s 21ms/step - loss: 0.3434 - accuracy: 0.8812 - val_loss: 0.4883 - val_accuracy: 0.8399
# Epoch 17/25
# 575/575 [==============================] - 14s 25ms/step - loss: 0.3369 - accuracy: 0.8825 - val_loss: 0.5514 - val_accuracy: 0.8320
# Epoch 18/25
# 575/575 [==============================] - 18s 32ms/step - loss: 0.3356 - accuracy: 0.8822 - val_loss: 0.4581 - val_accuracy: 0.8444
# Epoch 19/25
# 575/575 [==============================] - 18s 32ms/step - loss: 0.3202 - accuracy: 0.8886 - val_loss: 0.4045 - val_accuracy: 0.8736
# Epoch 20/25
# 575/575 [==============================] - 17s 30ms/step - loss: 0.3193 - accuracy: 0.8907 - val_loss: 0.5751 - val_accuracy: 0.8227
# Epoch 21/25
# 575/575 [==============================] - 19s 32ms/step - loss: 0.3090 - accuracy: 0.8912 - val_loss: 0.5398 - val_accuracy: 0.8250
# Epoch 22/25
# 575/575 [==============================] - 17s 29ms/step - loss: 0.2884 - accuracy: 0.9018 - val_loss: 0.6768 - val_accuracy: 0.7987
# Epoch 23/25
# 575/575 [==============================] - 16s 28ms/step - loss: 0.2898 - accuracy: 0.8982 - val_loss: 0.4119 - val_accuracy: 0.8678
# Epoch 24/25
# 575/575 [==============================] - 16s 27ms/step - loss: 0.2841 - accuracy: 0.9017 - val_loss: 0.3985 - val_accuracy: 0.8740
# Epoch 25/25
# 575/575 [==============================] - 17s 30ms/step - loss: 0.2762 - accuracy: 0.9053 - val_loss: 0.4010 - val_accuracy: 0.8744
# =============================================================================

# Plotting of model run

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Test Prediction, Change file path as necessary

prediction_path = "/content/drive/MyDrive/mosiac_virus.jpg"


img = tf.keras.utils.load_img(
    prediction_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

# =============================================================================
# 1/1 [==============================] - 0s 18ms/step
# This image most likely belongs to Septoria_leaf_spot with a 58.16 percent confidence.
# =============================================================================

# Test Prediction, Change file path as necessary

prediction_path = "/content/drive/MyDrive/mosiac_virus2.jpg"


img = tf.keras.utils.load_img(
    prediction_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

# =============================================================================
# 1/1 [==============================] - 0s 21ms/step
# This image most likely belongs to Tomato_Yellow_Leaf_Curl_Virus with a 62.00 percent confidence.
# =============================================================================


# Test Prediction, Change file path as necessary

prediction_path = "/content/drive/MyDrive/mosiac_virus3.jpg"


img = tf.keras.utils.load_img(
    prediction_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

# =============================================================================
# 1/1 [==============================] - 0s 20ms/step
# This image most likely belongs to Late_blight with a 59.69 percent confidence.
# =============================================================================

# Building model in Keras API


num_classes = len(class_names)

model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(256, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, name="outputs")
])

# Compiliation of model

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Print model summary

model.summary()

# =============================================================================
# Model: "sequential_19"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  sequential_17 (Sequential)  (None, 128, 128, 3)       0         
#                                                                  
#  rescaling_11 (Rescaling)    (None, 128, 128, 3)       0         
#                                                                  
#  conv2d_19 (Conv2D)          (None, 128, 128, 16)      448       
#                                                                  
#  max_pooling2d_19 (MaxPoolin  (None, 64, 64, 16)       0         
#  g2D)                                                            
#                                                                  
#  conv2d_20 (Conv2D)          (None, 64, 64, 32)        4640      
#                                                                  
#  max_pooling2d_20 (MaxPoolin  (None, 32, 32, 32)       0         
#  g2D)                                                            
#                                                                  
#  conv2d_21 (Conv2D)          (None, 32, 32, 64)        18496     
#                                                                  
#  max_pooling2d_21 (MaxPoolin  (None, 16, 16, 64)       0         
#  g2D)                                                            
#                                                                  
#  conv2d_22 (Conv2D)          (None, 16, 16, 128)       73856     
#                                                                  
#  max_pooling2d_22 (MaxPoolin  (None, 8, 8, 128)        0         
#  g2D)                                                            
#                                                                  
#  conv2d_23 (Conv2D)          (None, 8, 8, 256)         295168    
#                                                                  
#  max_pooling2d_23 (MaxPoolin  (None, 4, 4, 256)        0         
#  g2D)                                                            
#                                                                  
#  dropout_15 (Dropout)        (None, 4, 4, 256)         0         
#                                                                  
#  flatten_5 (Flatten)         (None, 4096)              0         
#                                                                  
#  dense_15 (Dense)            (None, 128)               524416    
#                                                                  
#  outputs (Dense)             (None, 11)                1419      
#                                                                  
# =================================================================
# Total params: 918,443
# Trainable params: 918,443
# Non-trainable params: 0
# _________________________________________________________________
# =============================================================================


# Model run, adjust epochs, callbacks and others as necessary 


epochs = 50
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=early_stop,
  steps_per_epoch=steps_per_epoch)

# =============================================================================
# Epoch 1/50
# 1149/1149 [==============================] - 36s 18ms/step - loss: 1.5607 - accuracy: 0.4402 - val_loss: 0.9303 - val_accuracy: 0.6675
# Epoch 2/50
# 1149/1149 [==============================] - 17s 14ms/step - loss: 0.8808 - accuracy: 0.6893 - val_loss: 0.6894 - val_accuracy: 0.7542
# Epoch 3/50
# 1149/1149 [==============================] - 16s 14ms/step - loss: 0.6819 - accuracy: 0.7605 - val_loss: 0.6874 - val_accuracy: 0.7592
# Epoch 4/50
# 1149/1149 [==============================] - 16s 14ms/step - loss: 0.5931 - accuracy: 0.7903 - val_loss: 0.5709 - val_accuracy: 0.7883
# Epoch 5/50
# 1149/1149 [==============================] - 17s 15ms/step - loss: 0.5401 - accuracy: 0.8097 - val_loss: 0.6908 - val_accuracy: 0.7513
# Epoch 6/50
# 1149/1149 [==============================] - 17s 14ms/step - loss: 0.4926 - accuracy: 0.8280 - val_loss: 0.5507 - val_accuracy: 0.8061
# Epoch 7/50
# 1149/1149 [==============================] - 16s 14ms/step - loss: 0.4442 - accuracy: 0.8433 - val_loss: 0.4931 - val_accuracy: 0.8326
# Epoch 8/50
# 1149/1149 [==============================] - 17s 14ms/step - loss: 0.4234 - accuracy: 0.8513 - val_loss: 0.4696 - val_accuracy: 0.8260
# Epoch 9/50
# 1149/1149 [==============================] - 17s 15ms/step - loss: 0.3838 - accuracy: 0.8654 - val_loss: 0.4431 - val_accuracy: 0.8475
# Epoch 10/50
# 1149/1149 [==============================] - 16s 14ms/step - loss: 0.3683 - accuracy: 0.8719 - val_loss: 0.5553 - val_accuracy: 0.8144
# Epoch 11/50
# 1149/1149 [==============================] - 16s 14ms/step - loss: 0.3609 - accuracy: 0.8730 - val_loss: 0.3653 - val_accuracy: 0.8730
# Epoch 12/50
# 1149/1149 [==============================] - 16s 14ms/step - loss: 0.3367 - accuracy: 0.8825 - val_loss: 0.4742 - val_accuracy: 0.8459
# Epoch 13/50
# 1149/1149 [==============================] - 16s 14ms/step - loss: 0.3146 - accuracy: 0.8886 - val_loss: 0.4523 - val_accuracy: 0.8425
# Epoch 14/50
# 1149/1149 [==============================] - 17s 14ms/step - loss: 0.3177 - accuracy: 0.8871 - val_loss: 0.4389 - val_accuracy: 0.8550
# Epoch 15/50
# 1149/1149 [==============================] - 17s 14ms/step - loss: 0.2986 - accuracy: 0.8971 - val_loss: 0.3439 - val_accuracy: 0.8752
# Epoch 16/50
# 1149/1149 [==============================] - 16s 14ms/step - loss: 0.2822 - accuracy: 0.9030 - val_loss: 0.3964 - val_accuracy: 0.8692
# Epoch 17/50
# 1149/1149 [==============================] - 16s 14ms/step - loss: 0.2723 - accuracy: 0.9062 - val_loss: 0.2551 - val_accuracy: 0.9088
# Epoch 18/50
# 1149/1149 [==============================] - 17s 15ms/step - loss: 0.2634 - accuracy: 0.9089 - val_loss: 0.3229 - val_accuracy: 0.8912
# Epoch 19/50
# 1149/1149 [==============================] - 16s 14ms/step - loss: 0.2550 - accuracy: 0.9126 - val_loss: 0.3126 - val_accuracy: 0.8914
# Epoch 20/50
# 1149/1149 [==============================] - 16s 14ms/step - loss: 0.2519 - accuracy: 0.9129 - val_loss: 0.3430 - val_accuracy: 0.8899
# Epoch 21/50
# 1149/1149 [==============================] - 17s 15ms/step - loss: 0.2480 - accuracy: 0.9166 - val_loss: 0.3184 - val_accuracy: 0.8945
# Epoch 22/50
# 1149/1149 [==============================] - 17s 15ms/step - loss: 0.2377 - accuracy: 0.9192 - val_loss: 0.2410 - val_accuracy: 0.9164
# Epoch 23/50
# 1149/1149 [==============================] - 16s 14ms/step - loss: 0.2411 - accuracy: 0.9180 - val_loss: 0.2883 - val_accuracy: 0.9050
# Epoch 24/50
# 1149/1149 [==============================] - 16s 14ms/step - loss: 0.2323 - accuracy: 0.9197 - val_loss: 0.3450 - val_accuracy: 0.8877
# Epoch 25/50
# 1149/1149 [==============================] - 17s 14ms/step - loss: 0.2254 - accuracy: 0.9213 - val_loss: 0.2732 - val_accuracy: 0.9110
# Epoch 26/50
# 1149/1149 [==============================] - 16s 14ms/step - loss: 0.2092 - accuracy: 0.9272 - val_loss: 0.2511 - val_accuracy: 0.9189
# Epoch 27/50
# 1149/1149 [==============================] - 16s 14ms/step - loss: 0.2113 - accuracy: 0.9291 - val_loss: 0.4100 - val_accuracy: 0.8783
# Epoch 28/50
# 1149/1149 [==============================] - 17s 15ms/step - loss: 0.2040 - accuracy: 0.9310 - val_loss: 0.2440 - val_accuracy: 0.9226
# Epoch 29/50
# 1149/1149 [==============================] - 17s 14ms/step - loss: 0.1935 - accuracy: 0.9326 - val_loss: 0.4428 - val_accuracy: 0.8740
# Epoch 30/50
# 1149/1149 [==============================] - 17s 15ms/step - loss: 0.1967 - accuracy: 0.9356 - val_loss: 0.4069 - val_accuracy: 0.8682
# Epoch 31/50
# 1149/1149 [==============================] - 16s 14ms/step - loss: 0.2031 - accuracy: 0.9313 - val_loss: 0.2651 - val_accuracy: 0.9185
# Epoch 32/50
# 1149/1149 [==============================] - 16s 14ms/step - loss: 0.1840 - accuracy: 0.9373 - val_loss: 0.2516 - val_accuracy: 0.9245
# =============================================================================


# Plotting of model run


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = history.epoch
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Test Prediction, Change file path as necessary

prediction_path = "/content/drive/MyDrive/mosiac_virus.jpg"


img = tf.keras.utils.load_img(
    prediction_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

# =============================================================================
# 1/1 [==============================] - 0s 100ms/step
# This image most likely belongs to Tomato_mosaic_virus with a 54.39 percent confidence.
# =============================================================================

# Test Prediction, Change file path as necessary

prediction_path = "/content/drive/MyDrive/mosiac_virus2.jpg"


img = tf.keras.utils.load_img(
    prediction_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

# =============================================================================
# 1/1 [==============================] - 0s 19ms/step
# This image most likely belongs to Tomato_mosaic_virus with a 89.75 percent confidence.
# =============================================================================


# Test Prediction, Change file path as necessary

prediction_path = "/content/drive/MyDrive/mosiac_virus3.jpg"


img = tf.keras.utils.load_img(
    prediction_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

# =============================================================================
# 1/1 [==============================] - 0s 22ms/step
# This image most likely belongs to Bacterial_spot with a 51.33 percent confidence.
# =============================================================================


# Conversion for TF Lite

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)












