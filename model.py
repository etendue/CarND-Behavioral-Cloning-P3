import pandas as pd
import numpy as np
import cv2

# load the inputs using pandas
# read "driving_log.csv" file to get the image paths and steering angle
column_names = ["center_img_path", "left_img_path", "right_img_path", "steering_angle"]
df = pd.read_csv("data/driving_log.csv", header=None, usecols=[0, 1, 2, 3], names=column_names)

# adjust the file path by trim unnecessary path
from os.path import basename

df["center_img_path"] = df["center_img_path"].map(lambda x: "data/IMG/{}".format(basename(x)))
df["left_img_path"] = df["left_img_path"].map(lambda x: "data/IMG/{}".format(basename(x)))
df["right_img_path"] = df["right_img_path"].map(lambda x: "data/IMG/{}".format(basename(x)))

# TODO: augment data here
# add a column to indicate whether the image needs to be flipped
df["flip"] = False
samples = df[["center_img_path", "steering_angle","flip"]].values

# flipped samples are only the flag "flip" is True
df["flip"] = True
flipped_samples = df[["center_img_path", "steering_angle","flip"]].values

# use left camera images
df["flip"] = False
df["steering_angle"] += 0.2
left_samples = df[["left_img_path", "steering_angle","flip"]].values

# make a deep copy of data

# use right camera images
df["steering_angle"] -= 0.2
right_samples = df[["right_img_path", "steering_angle","flip"]]

total_data = np.concatenate((samples,flipped_samples,left_samples,right_samples),axis = 0)

batch_size = 32


def generator(data, batch_size=32):
    num_samples = len(data)
    batch_num = num_samples // batch_size
    while 1:  # epoch
        np.random.shuffle(data)
        for i in range(batch_num):  # batch
            batch_data = data[i * batch_size: (i + 1) * batch_size]
            images = []
            angles = []
            for b in batch_data:
                path  = b[0]
                angle = b[1]
                flip  = b[2]
                image = cv2.imread(path)

                # whether flip the image
                if flip:
                    image = cv2.flip(image,flipCode=1)
                    angle = -angle

                images.append(image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield (X_train, y_train)


from sklearn.model_selection import train_test_split

train_samples, val_samples = train_test_split(total_data, test_size=0.2)

train_generator = generator(train_samples, batch_size=batch_size)
val_generator = generator(val_samples, batch_size=batch_size)

# construct neural network model here
from keras.layers import Dense, Conv2D, Input, Lambda, Dropout, Cropping2D, Flatten
from keras.models import Sequential
from keras.callbacks import EarlyStopping

# use network published by Nvidia TODO:add link here
model = Sequential()

# 1. Pre-process the inputs
# 1.1 Crop the image
input_shape = (160, 320, 3)
# cropped_shape = (90,320,3)

model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=input_shape))

# 1.2 Layer1: Normalize input using Lambda layer
model.add(Lambda(lambda x: x / 127.5 - 1.0))

# 2. add convolution layers
# 2.1 Layer 2: Convolution 5x5 kernel with 2x2 stride
model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'))

# 2.2 Layer 3: Convolution 5x5 kernel with 2x2 stride
model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'))

# 2.3 Layer 4: Convolution 5x5 kernel with 2x2 stride
model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'))

# 2.4 Layer 5: Convolution 3x3 kernel with 1x1 stride
model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))

# 2.5 Layer 6: Convolution 3x3 kernel with 1x1 stride
model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))

# 2.6 Layer 7: Flatten
model.add(Flatten())

# 2.7 Layer 8: Fully Connected Layer
model.add(Dense(1024))

# 2.8 Layer 9: Dropout
model.add(Dropout(0.5))

# 2.9 Layer 10: Fully Connected Layer
model.add(Dense(64))

# 2.10 Layer 11: Fully Connected Layer
model.add(Dense(1))

# Train the model
model.compile(loss='mse', optimizer='adam')

train_steps = len(train_samples) // batch_size
val_steps = len(val_samples) // batch_size

# apply early stop
early_stop = EarlyStopping(monitor='val_loss', patience=2)
history = model.fit_generator(train_generator, steps_per_epoch=train_steps, \
                              epochs=20, verbose=1, callbacks=[early_stop], \
                              validation_data=val_generator, validation_steps=val_steps)

model.save('model.h5')
exit()