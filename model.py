import pandas as pd
import numpy as np
import cv2
from sklearn.utils import shuffle
from scipy import interpolate
from os.path import basename


def prepare_data():
    # read csv file
    # data set from udacity sample data
    df_center = pd.read_csv("data0/driving_log.csv", header=None, usecols=[0, 3], names=["path", "angle"])
    df_left = pd.read_csv("data0/driving_log.csv", header=None, usecols=[1, 3], names=["path", "angle"])
    df_right = pd.read_csv("data0/driving_log.csv", header=None, usecols=[2, 3], names=["path", "angle"])
    df_center["path"] = df_center["path"].map(lambda x: "data0/IMG/{}".format(basename(x)))
    df_left["path"] = df_left["path"] .map(lambda x: "data0/IMG/{}".format(basename(x)))
    df_right["path"] = df_right["path"] .map(lambda x: "data0/IMG/{}".format(basename(x)))

    # additional data collected to compensate the bridge scene
    df_more =  pd.read_csv("data_add/driving_log.csv", header=None, usecols=[0, 3], names=["path", "angle"])
    df_more["path"] = df_more["path"].map(lambda x: "data_add/IMG/{}".format(basename(x)))
    df_more["flip"] = False

    # add correction, using left and right cameras
    df_left["angle"] += 0.2
    df_right["angle"] -= 0.2
    df = pd.concat((df_center, df_left, df_right,df_more))
    # add a flag to indicate the image is flipped
    df["flip"] = False
    # make a copy for flip images
    # augment data by flipping images

    df_flip = df.copy()
    df_flip["flip"] = True
    df_flip["angle"] = -df["angle"]
    # combine the data set
    df = pd.concat((df, df_flip))

    # shuffle the data
    df = shuffle(df)
    df = df.reset_index(drop=True)

    # analyze data
    n, bins = np.histogram(df["angle"], bins=200)

    # data distribution
    distrib = pd.DataFrame(data={"x": bins[:-1], "y": n})
    # remove 0 bins

    # there are groups which contains >8000 samples within 40 000 samples in total
    # this is mainly the samples with 0 angle, and those augmented by introducing left, right cameras
    # i.e. angle ~ 0.2 ~-0.2
    distrib = distrib[distrib.y != 0]
    x = distrib[distrib.y < 2000]["x"]
    y = distrib[distrib.y < 2000]["y"]

    # fit distribution in order to remove spikes
    f = interpolate.interp1d(x, y)
    # make reasonable counts for spikes
    y_prime = f(distrib[distrib.y > 2000]["x"])

    # overwrite original data
    row_index = distrib.y > 2000
    distrib.loc[row_index, "y"] = y_prime

    # group the data with angle range .i.e. bins
    group_index = np.digitize(df["angle"], bins)
    group_by = df.groupby(group_index)

    # sample the data
    samples = []
    # add additional data
    samples.extend(df_more[["path", "angle", "flip"]].values)

    for (k, g), y in zip(group_by, distrib["y"]):
        sample = g[["path", "angle", "flip"]].values
        count = int(y)
        samples.extend(sample[:count])

    np.random.shuffle(samples)

    return np.array(samples)


total_data = prepare_data()

batch_size = 64

# use generator to generate batch data for training
def generator(data, batchsize=32):
    num_samples = len(data)
    batch_num = num_samples // batchsize
    while 1:  # epoch
        np.random.shuffle(data)
        for i in range(batch_num):  # batch
            batch_data = data[i * batchsize: (i + 1) * batchsize]
            images = []
            angles = []
            for b in batch_data:
                path = b[0]
                angle = b[1]
                flip = b[2]
                image = cv2.imread(path)

                # whether flip the image
                if flip:
                    image = cv2.flip(image, flipCode=1)

                images.append(image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield (X_train, y_train)


from sklearn.model_selection import train_test_split
# split the train, validation data set
# no test data set is used, as there is no clear criteria to check quality with test data
train_samples, val_samples = train_test_split(total_data, test_size=0.2)

train_generator = generator(train_samples, batchsize=batch_size)
val_generator = generator(val_samples, batchsize=batch_size)

# construct neural network model here
from keras.layers import Dense, Conv2D, Input, Lambda, Dropout, Cropping2D, Flatten
from keras.models import Sequential
from keras.callbacks import EarlyStopping

# use network published by Nvidia
# http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

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
model.add(Conv2D(24, 5, 5, subsample=(2, 2), border_mode='valid', activation='relu'))

# 2.2 Layer 3: Convolution 5x5 kernel with 2x2 stride
model.add(Conv2D(36, 5, 5, subsample=(2, 2), border_mode='valid', activation='relu'))

# 2.3 Layer 4: Convolution 5x5 kernel with 2x2 stride
model.add(Conv2D(48, 5, 5, subsample=(2, 2), border_mode='valid', activation='relu'))

# 2.4 Layer 5: Convolution 3x3 kernel with 1x1 stride
model.add(Conv2D(64, 3, 3, subsample=(1, 1), border_mode='valid', activation='relu'))

# 2.5 Layer 6: Convolution 3x3 kernel with 1x1 stride
model.add(Conv2D(64, 3, 3, subsample=(1, 1), border_mode='valid', activation='relu'))

# 2.6 Layer 7: Flatten
model.add(Flatten())

# 2.7 Layer 8: Fully Connected Layer
model.add(Dense(256))

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
history = model.fit_generator(train_generator, samples_per_epoch=train_steps * batch_size, \
                              nb_epoch=20, verbose=1, callbacks=[early_stop], \
                              validation_data=val_generator, nb_val_samples=val_steps * batch_size)

model.save('model.h5')
exit()
