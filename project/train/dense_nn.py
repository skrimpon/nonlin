import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, Normalizer, PowerTransformer
from sklearn.model_selection import train_test_split
from tensorflow.keras import losses

import matplotlib.pyplot as plt

# Print tensorflow version. This code has been tested with 2.3.0
print(f'Tensorflow Version: {tf.__version__}')
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

nit = 10  # num of iterations
nrx = 16  # num of receiver antennas
nsnr = 29  # num of snr points
nx = 10000  # num of tx samples


def make_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            units=32,
            use_bias=True,
            activation='softsign',
            bias_regularizer=None,
            kernel_regularizer=None,
            bias_initializer='lecun_uniform',
            kernel_initializer='lecun_uniform'
        ),
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
            units=128,
            use_bias=True,
            activation='softsign',
            bias_regularizer=None,
            kernel_regularizer=None,
            bias_initializer='lecun_uniform',
            kernel_initializer='lecun_uniform'
        ),
        tf.keras.layers.Dense(
            units=256,
            use_bias=True,
            activation='softsign',
            bias_regularizer=None,
            kernel_regularizer=None,
            bias_initializer='lecun_uniform',
            kernel_initializer='lecun_uniform'
        ),
        tf.keras.layers.Dense(
            units=512,
            use_bias=True,
            activation='softsign',
            bias_regularizer=None,
            kernel_regularizer=None,
            bias_initializer='lecun_uniform',
            kernel_initializer='lecun_uniform'
        ),
        tf.keras.layers.Dense(
            units=1024,
            use_bias=True,
            activation='softsign',
            bias_regularizer=None,
            kernel_regularizer=None,
            bias_initializer='lecun_uniform',
            kernel_initializer='lecun_uniform'
        ),
        tf.keras.layers.Dense(
            units=512,
            use_bias=True,
            activation='softsign',
            bias_regularizer=None,
            kernel_regularizer=None,
            bias_initializer='lecun_uniform',
            kernel_initializer='lecun_uniform'
        ),
        tf.keras.layers.Dense(
            units=256,
            use_bias=True,
            activation='softsign',
            bias_regularizer=None,
            kernel_regularizer=None,
            bias_initializer='lecun_uniform',
            kernel_initializer='lecun_uniform'
        ),
        tf.keras.layers.Dense(
            units=128,
            use_bias=True,
            activation='softsign',
            bias_regularizer=None,
            kernel_regularizer=None,
            bias_initializer='lecun_uniform',
            kernel_initializer='lecun_uniform'
        ),
        tf.keras.layers.Dense(
            units=64,
            use_bias=True,
            activation='softsign',
            bias_regularizer=None,
            kernel_regularizer=None,
            bias_initializer='lecun_uniform',
            kernel_initializer='lecun_uniform'
        ),
        tf.keras.layers.Dense(32, activation='linear')
    ])

    adam_opt = tf.keras.optimizers.Adam(
        learning_rate=1e-3,
        amsgrad=False,
        epsilon=1e-7
    )

    model.compile(
        optimizer=adam_opt,
        loss=losses.MeanSquaredError(),
    )

    return model


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


def snr(Phi, x, w):
    xh = sum(Phi.T * np.conj(w), 0) / np.sum(np.abs(w) ** 2, 0)
    a = np.mean(np.conj(xh) * x) / np.mean(np.abs(x) ** 2)
    d_var = np.mean(np.abs(xh - a * x) ** 2)
    snr_out = 10 * np.log10(np.abs(a) ** 2 / d_var)
    return snr_out


def my_norm(A):
    A_min = np.min(A)
    A_max = np.max(A)
    A -= A_min
    A /= (A_max - A_min)
    return A


def test(model, y_rffe, pwr_out, x, w, it, isnr_tst):
    pin = np.array((
        -130.4307, -125.4307, -120.4307, -115.4307, -110.4307, -105.4307, -100.4307, -95.4307,
        -90.4307, -85.4307, -80.4307, -75.4307, -70.4307, -65.4307, -60.4307, -55.4307,
        -50.4307, -45.4307, -40.4307, -35.4307, -30.4307, -25.4307, -20.4307, -15.4307,
        -10.4307, -5.4307, -0.4307, 4.5693, 9.5693))

    pred_snr = np.zeros(nsnr)
    base_snr = np.zeros(nsnr)

    for isnr in range(nsnr):
        X = np.hstack((y_rffe[:, isnr, :].real, y_rffe[:, isnr, :].imag))

        X_min = np.min(X)
        X_max = np.max(X)
        X -= X_min
        X /= (X_max - X_min)
        X = np.hstack((X, 10 ** (0.1 * (pwr_out[:, isnr, :] - 30))))

        pred = model(X).numpy()

        pred *= (X_max - X_min)
        pred += X_min

        pred = pred[:, :16] + 1j * pred[:, 16:]
        pred_snr[isnr] = snr(pred, x, w)
        base_snr[isnr] = snr(y_rffe[:, isnr, :], x, w)

    plt.title('Iter: '+str(it)+' SNR: '+str(isnr_tst))
    plt.plot(pin, base_snr, 'bs')
    plt.plot(pin, pred_snr, 'rd')
    plt.grid()
    plt.xlabel('Receive power per antenna [dBm]')
    plt.ylabel('Output SNR $\;(\gamma_\mathrm{out})\;$ [dB]')
    plt.legend(['Reference', 'DNN'])
    plt.show()


model = make_model()
for it in range(2):
    [x, w, y_ant, y_rffe, pwr_out] = parse_file(it)
    test(model, y_rffe, pwr_out, x, w, it, -1)
    for it2 in range(2):
        for isnr in range(nsnr - 1, 0, -1):
            print(f'{it}\t{it2}\t{isnr}')
            r = np.hstack((y_ant[:, isnr, :].real, y_ant[:, isnr, :].imag))
            X = np.hstack((y_rffe[:, isnr, :].real, y_rffe[:, isnr, :].imag))
            r = my_norm(r)
            X = my_norm(X)
            X = np.hstack((X, 10 ** (0.1 * (pwr_out[:, isnr, :] - 30))))

            x_train, x_test, y_train, y_test = train_test_split(X, r, shuffle=True, test_size=0.1)

            model.fit(x_train, y_train,
                      epochs=10,
                      batch_size=64,
                      shuffle=False,
                      validation_data=(x_test, y_test),
                      verbose=False)
            test(model, y_rffe, pwr_out, x, w, it, isnr)
