import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten

import cv2, os
import numpy as np
import matplotlib.image as mpimg



IMAGE_HEIGHT = 66
IMAGE_WIDTH = 200
IMAGE_CHANNELS = 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def preprocess(image):
    # Crop the image (removing the sky at the top and the car front at the bottom)
    image = image[60:-25, :, :]
    # Resize the image to the input shape used by the network model
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
    # Convert the image from RGB to YUV (This is what the NVIDIA model does)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image


def random_image(data_dir, center, left, right, steering_angle):
    choice = np.random.choice(3)
    if choice == 0:
        left_img = mpimg.imread(os.path.join(data_dir, left.strip()))
        return left_img, steering_angle + 0.2
    elif choice == 1:
        right_img = mpimg.imread(os.path.join(data_dir, right.strip()))
        return right_img, steering_angle - 0.2
    center_img = mpimg.imread(os.path.join(data_dir, center.strip()))
    return center_img, steering_angle


def random_flip(image, steering_angle):

    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


def random_translate(image, steering_angle, range_x, range_y):

    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle



def random_brightness(image):

    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augument(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):

    image, steering_angle = random_image(data_dir, center, left, right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    image = random_brightness(image)
    return image, steering_angle


def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):

    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]
            # argumentation
            if is_training and np.random.rand() < 0.6:
                image, steering_angle = augument(data_dir, center, left, right, steering_angle)
            else:
                image = mpimg.imread(os.path.join(data_dir, center.strip()))
            # add the image and steering angle to the batch
            images[i] = preprocess(image)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers


from workspace_utils import active_session


if __name__ == '__main__':
    data_dir = 'data'
    learning_rate = 1.0e-4
    batch_size = 40
    samples_per_epoch = 20000
    nb_epoch = 10
    keep_prob = 0.5
    # load data

    data_df = pd.read_csv(data_dir + '/driving_log.csv')

    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only='true',
                                 mode='auto')


    # create model
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    # train model
    with active_session():
        model.compile(loss='mean_squared_error',
                      optimizer=Adam(lr=learning_rate))

        model.fit_generator(batch_generator(data_dir, X_train, y_train, batch_size, True),
                            samples_per_epoch,
                            nb_epoch,
                            max_q_size=1,
                            validation_data=batch_generator(data_dir, X_valid, y_valid, batch_size, False),
                            nb_val_samples=len(X_valid),
                            callbacks=[checkpoint],
                            verbose=1)



