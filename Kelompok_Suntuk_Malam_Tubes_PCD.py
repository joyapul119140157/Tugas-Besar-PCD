#library: tensorflow, keras, numpy, cv2, os, matplotlib
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

img_width, img_height = 150, 150
batch_size = 32
epochs = 10
train_data_dir = './train'
validation_data_dir = './val'

# Preprocess image
def preprocess_image(image):
    
    image = image / 255.0
    
    # Convert to grayscale
    gray_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Compute Sobel
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
    combined_sobel = np.sqrt(sobelx**2 + sobely**2)
    # Normalize
    combined_sobel = combined_sobel / np.max(combined_sobel)
    # Convert to 8-bit
    sobel_8u = (combined_sobel * 255).astype(np.uint8)
    # Threshold
    _, thresholded_sobel = cv2.threshold(sobel_8u, 50, 255, cv2.THRESH_BINARY)
    # Dilate and erode
    kernel = np.ones((3, 3), np.uint8)
    dilated_sobel = cv2.dilate(thresholded_sobel, kernel, iterations=1)
    eroded_sobel = cv2.erode(dilated_sobel, kernel, iterations=1)
    # Combine Sobel with original image
    combined_sobel_rgb = cv2.cvtColor(eroded_sobel, cv2.COLOR_GRAY2RGB)
    # Final image
    final_image = np.clip(image + combined_sobel_rgb / 255.0, 0, 1)

    return final_image

# Data generators with preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_image
)

val_datagen = ImageDataGenerator(
    rescale=1.0/255,
    preprocessing_function=preprocess_image
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# Model CNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Adjusting steps_per_epoch and validation_steps
steps_per_epoch = int(np.ceil(train_generator.samples / batch_size))
validation_steps = int(np.ceil(validation_generator.samples / batch_size))

# Train model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    epochs=epochs
)

# Save model
model.save('wrinkle_detection_model.h5')

# Plot accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
