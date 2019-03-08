import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten
from utils import INPUT_SHAPE, batch_generator


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
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))

        model.fit_generator(batch_generator(data_dir, X_train, y_train, batch_size, True),
                            samples_per_epoch,
                            nb_epoch,
                            max_q_size=1,
                            validation_data=batch_generator(data_dir, X_valid, y_valid, batch_size, False),
                            nb_val_samples=len(X_valid),
                            callbacks=[checkpoint],
                            verbose=1)



