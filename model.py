import os
import csv
import cv2
import numpy as np
import sklearn
import math
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.models import Model

# Reading model csv file generated from training laps
samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for sample in reader:
        samples.append(sample)

# Split our data to train and validation with 0.2
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Define a generator to read our csv files parameters from samples array
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = 'data/IMG/'+batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    images.append(image)

                correction = 0.2
                angle = float(batch_sample[3])
                angles.append(angle)
                angles.append(angle+correction)
                angles.append(angle-correction)

            augmented_images, augmented_angles = [], []
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image,1))
                augmented_angles.append(angle*-1.0)

            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)

            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


model = Sequential()
model.add(Lambda(lambda x: x/255 - 0.5,
        input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24,5, strides=(2,2),activation='relu'))
model.add(Conv2D(36,5, strides=(2,2),activation='relu'))
model.add(Conv2D(48,5, strides=(2,2),activation='relu'))
model.add(Conv2D(64,3,activation='relu'))
model.add(Conv2D(64,3,activation='relu'))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

BATCH_SIZE = 32

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,
            steps_per_epoch= math.ceil(len(train_samples) / BATCH_SIZE),
            epochs=3,
            verbose=1,
            validation_data=validation_generator,
            validation_steps=math.ceil(len(validation_samples) / BATCH_SIZE))

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

# Save our model to model.h5
model.save('model.h5')
