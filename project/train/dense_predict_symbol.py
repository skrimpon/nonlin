import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, Normalizer, PowerTransformer
from sklearn.model_selection import train_test_split
from tensorflow.keras import losses
from tensorflow import keras
import matplotlib.pyplot as plt
import datetime

# Print tensorflow version. This code has been tested with 2.3.0
print(f'Tensorflow Version: {tf.__version__}')
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

nit = 5  # num of iterations
nrx = 16  # num of receiver antennas
nsnr = 22  # num of snr points
nx = 10000  # num of tx samples

def parse_file(it):
    df = pd.read_csv(r'../../datasets/new/dataset_' + str(it+1) + '.csv')

    x = np.char.replace(np.array(df['x'], dtype=str), 'i', 'j').astype(np.complex)

    w = np.array([np.char.replace(np.array(df['w_' + str(i + 1)], dtype=str), 'i', 'j').astype(np.complex)
                  for i in range(nrx)], dtype=complex)

    y_ant = np.array([
        np.char.replace(np.array(df['yant_' + str(isnr * nrx + irx + 1)], dtype=str), 'i', 'j').astype(np.complex)
        for isnr in range(nsnr) for irx in range(nrx)
    ], dtype=complex).T.reshape(nx, nsnr, nrx)

    y_rffe = np.array([
        np.char.replace(np.array(df['yrffe_' + str(isnr * nrx + irx + 1)], dtype=str), 'i', 'j').astype(np.complex)
        for isnr in range(nsnr) for irx in range(nrx)
    ], dtype=complex).T.reshape(nx, nsnr, nrx)

    pwr_out = np.array([
        np.char.replace(np.array(df['pwrOut_' + str(isnr * nrx + irx + 1)], dtype=str), 'i', 'j').astype(np.float)
        for isnr in range(nsnr) for irx in range(nrx)
    ], dtype=float).T.reshape(nx, nsnr, nrx)

    return [x, w, y_ant, y_rffe, pwr_out]



def parse_multiple_files(it_list):
    import multiprocessing as mp
    with mp.Pool(8) as p:
        uu=p.map(parse_file,it_list)
    # num_its = len(it_list)
    # uu = [parse_file(_) for _ in it_list]
    x       = np.concatenate([i[0] for i in uu],axis=0)
    w       = np.concatenate([i[1] for i in uu],axis=1)
    y_ant   = np.concatenate([i[2] for i in uu],axis=0)
    y_rffe  = np.concatenate([i[3] for i in uu], axis=0)
    pwr_out  = np.concatenate([i[4] for i in uu], axis=0)
    print('Training on {} samples'.format(y_rffe.shape[0]))
    return [x, w, y_ant, y_rffe, pwr_out]


#
# def make_model(input_shape):
#     model = keras.Sequential()
#     # Input
#     model.add(keras.Input(shape=(input_shape,)))
#     model.add(keras.layers.BatchNormalization())
#     # Dense 1
#     model.add(keras.layers.Dense(2048))
#     model.add(keras.layers.Dropout(0.3))
#     model.add(keras.layers.BatchNormalization())
#     # Dense 2
#     model.add(keras.layers.Dense(1024))
#     model.add(keras.layers.Dropout(0.3))
#     model.add(keras.layers.BatchNormalization())
#     # Dense 3
#     model.add(keras.layers.Dense(512))
#     model.add(keras.layers.Dropout(0.3))
#     model.add(keras.layers.BatchNormalization())
#     # model.add(keras.layers.LeakyReLU())
#     # Dense 4
#     model.add(keras.layers.Dense(256))
#     model.add(keras.layers.Dropout(0.3))
#     model.add(keras.layers.BatchNormalization())
#     # model.add(keras.layers.LeakyReLU())
#     # Dense 4
#     model.add(keras.layers.Dense(2))
#     return model


def make_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            units=128,
            use_bias=True,
            activation='softsign',
            bias_regularizer=None,
            kernel_regularizer=None,
            bias_initializer='lecun_uniform',
            kernel_initializer='lecun_uniform'
        ),
        # tf.keras.layers.Dropout(0.1),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(
            units=256,
            use_bias=True,
            activation='softsign',
            bias_regularizer=None,
            kernel_regularizer=None,
            bias_initializer='lecun_uniform',
            kernel_initializer='lecun_uniform'
        ),
        # tf.keras.layers.Dropout(0.1),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(
            units=256,
            use_bias=True,
            activation='softsign',
            bias_regularizer=None,
            kernel_regularizer=None,
            bias_initializer='lecun_uniform',
            kernel_initializer='lecun_uniform'
        ),
        # tf.keras.layers.Dropout(0.1),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(
            units=256,
            use_bias=True,
            activation='softsign',
            bias_regularizer=None,
            kernel_regularizer=None,
            bias_initializer='lecun_uniform',
            kernel_initializer='lecun_uniform'
        ),
        # tf.keras.layers.Dropout(0.1),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(
            units=512,
            use_bias=True,
            activation='softsign',
            bias_regularizer=None,
            kernel_regularizer=None,
            bias_initializer='lecun_uniform',
            kernel_initializer='lecun_uniform'
        ),
        # tf.keras.layers.Dropout(0.1),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(
            units=1024,
            use_bias=True,
            activation='softsign',
            bias_regularizer=None,
            kernel_regularizer=None,
            bias_initializer='lecun_uniform',
            kernel_initializer='lecun_uniform'
        ),
        # tf.keras.layers.Dropout(0.1),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(
            units=512,
            use_bias=True,
            activation='softsign',
            bias_regularizer=None,
            kernel_regularizer=None,
            bias_initializer='lecun_uniform',
            kernel_initializer='lecun_uniform'
        ),
        # tf.keras.layers.Dropout(0.1),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(
            units=256,
            use_bias=True,
            activation='softsign',
            bias_regularizer=None,
            kernel_regularizer=None,
            bias_initializer='lecun_uniform',
            kernel_initializer='lecun_uniform'
        ),
        tf.keras.layers.Dropout(0.1),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(
            units=128,
            use_bias=True,
            activation='softsign',
            bias_regularizer=None,
            kernel_regularizer=None,
            bias_initializer='lecun_uniform',
            kernel_initializer='lecun_uniform'
        ),
        # tf.keras.layers.Dropout(0.1),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(
            units=64,
            use_bias=True,
            activation='softsign',
            bias_regularizer=None,
            kernel_regularizer=None,
            bias_initializer='lecun_uniform',
            kernel_initializer='lecun_uniform'
        ),
        tf.keras.layers.Dense(
            units=32,
            use_bias=True,
            activation='linear',
            bias_regularizer=None,
            kernel_regularizer=None,
            bias_initializer='lecun_uniform',
            kernel_initializer='lecun_uniform'
        ),
        tf.keras.layers.Dense(2, activation='linear')
    ])

    adam_opt = tf.keras.optimizers.Adam(
        learning_rate=1e-3,
        amsgrad=False,
        epsilon=1e-7
    )

    sgd_opt = tf.keras.optimizers.SGD()

    model.compile(
        optimizer=adam_opt,
        loss=losses.MeanSquaredError(),
    )

    return model


def snr(pred, x_orig):
    xh = pred
    a = np.mean(np.conj(xh) * x_orig) / np.mean(np.abs(x_orig) ** 2)
    d_var = np.mean(np.abs(xh - a * x_orig) ** 2)
    snr_out = 10 * np.log10(np.abs(a) ** 2 / d_var)
    return snr_out


def test(model, x_test, y_test):
    pred_snr = np.zeros(nsnr)
    for isnr in range(nsnr):
        x_test_isnr = x_test[:, isnr, :]
        y_test_isnr = y_test[:, isnr, :]
        x_orig = y_test_isnr[:,0] + 1j * y_test_isnr[:,1]
        pred = model(x_test_isnr).numpy()
        pred = pred[:,0] + 1j*pred[:,1]
        pred_snr[isnr] = snr(pred, x_orig)
    return pred_snr


MODEL_BASENAME = 'DENSE_symbol_softsing'
model_snr_recordings_file = './model_checkpoints/' + MODEL_BASENAME + '_snr_recordings.txt'


if __name__ == '__main__':
    it_list_train = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] #,8,9,10,11,12,13,14,15
    [x, w, y_ant, y_rffe, pwr_out] = parse_multiple_files(it_list_train)
    model = make_model()
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    w_in = w.transpose()
    X_all = np.concatenate([y_rffe.real, y_rffe.imag], axis=2)
    X_min = X_all.min(axis=2).min(axis=0)
    X_max = X_all.max(axis=2).max(axis=0)
    X_all = X_all - X_min[np.newaxis,:,np.newaxis]
    X_all = X_all / (X_max[np.newaxis,:,np.newaxis] - X_min[np.newaxis,:,np.newaxis])
    w_in = (w_in + 1) / 2
    w_in = np.repeat(w_in[:,np.newaxis,:], X_all.shape[1], axis=1)
    my_pwr = 10 ** (0.1 * (pwr_out - 30))
    X_all = np.concatenate([X_all,w_in.real,w_in.imag,my_pwr], axis=2)
    r_all = np.repeat(np.concatenate([x[:,np.newaxis].real,x[:,np.newaxis].imag],axis=1)[:,np.newaxis,:], X_all.shape[1], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(X_all, r_all, shuffle=True, test_size=0.1)

    with open(model_snr_recordings_file, 'w') as f:
        f.write('ModelName,')
        for isnr in range(nsnr-1):
            f.write('snr_{},'.format(isnr))
            # print('snr_{},'.format(isnr))
        f.write('snr_{}\n'.format(nsnr-1))
    # Train different datasets over the same model
    it=1
    for it2 in range(5):
        for isnr in range(nsnr - 1, 0, -1):
                iteration_string = '{}_{}_{}'.format(it, it2, isnr)
                print(iteration_string)
                model.fit(x_train[:, isnr, :], y_train[:, isnr, :], epochs=10, batch_size=64, shuffle=False,
                          validation_data=(x_test[:, isnr, :], y_test[:, isnr, :]), verbose=2,
                          callbacks=[tensorboard_callback])
                model_string = './model_checkpoints/' + MODEL_BASENAME + '_' + iteration_string
                model_name = MODEL_BASENAME + '_' + iteration_string
                model.save(model_string)
                pred_snr = test(model, x_test, y_test)
                with open(model_snr_recordings_file, 'a') as f:
                    f.write('{},'.format(model_name))
                    for jsnr in range(nsnr - 1):
                        f.write('{},'.format(pred_snr[jsnr]))
                    f.write('{}\n'.format(pred_snr[nsnr-1]))

# Use this after training to load a best model for an snr, find best model from model_snr_recordings file for each snr
# vv = tf.keras.models.load_model('./model_checkpoints/dense_model_1_2_19')


#
# for it in range(2):
#     [x, w, y_ant, y_rffe, pwr_out] = parse_file(it)
#     # test(model, y_rffe, pwr_out, x, w, it, -1)
#     # Train the same dataset over the same model multiple times.
#     for it2 in range(2):
#         for isnr in range(nsnr - 1, 0, -1):
#             print(f'{it}\t{it2}\t{isnr}')
#             r = np.hstack((y_ant[:, isnr, :].real, y_ant[:, isnr, :].imag))
#             X = np.hstack((y_rffe[:, isnr, :].real, y_rffe[:, isnr, :].imag))
#             r = my_norm(r)
#             X = my_norm(X)
#             X = np.hstack((X, 10 ** (0.1 * (pwr_out[:, isnr, :] - 30))))
#
#             x_train, x_test, y_train, y_test = train_test_split(X, r, shuffle=True, test_size=0.1)
#
#             model.fit(x_train, y_train,
#                       epochs=10,
#                       batch_size=64,
#                       shuffle=False,
#                       validation_data=(x_test, y_test),
#                       verbose=False)
#             test(model, y_rffe, pwr_out, x, w, it, isnr)


