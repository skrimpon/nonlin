import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, Normalizer, PowerTransformer
from sklearn.model_selection import train_test_split
from tensorflow.keras import losses
from tensorflow import keras
import matplotlib.pyplot as plt
import datetime
import pandas as pd

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
    x       = np.concatenate([i[0] for i in uu],axis=0)
    w       = np.concatenate([i[1] for i in uu],axis=1)
    y_ant   = np.concatenate([i[2] for i in uu],axis=0)
    y_rffe  = np.concatenate([i[3] for i in uu], axis=0)
    pwr_out  = np.concatenate([i[4] for i in uu], axis=0)
    print('Training on {} samples'.format(y_rffe.shape[0]))
    return [x, w, y_ant, y_rffe, pwr_out]


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
        -91.5771, -81.5771, -71.5771, -61.5771, -51.5771,
        -41.5771, -31.5771, -29.5771, -27.5771, -25.5771,
        -23.5771, -21.5771, -19.5771, -17.5771, -15.5771,
        -13.5771, -11.5771, -9.5771, -7.5771, -5.5771,
        -3.5771, -1.5771))

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

    return pred_snr


MODEL_BASENAME = 'dense_model_test_sep'
model_snr_recordings_file = './model_checkpoints/' + MODEL_BASENAME + '_snr_recordings.txt'


if __name__ == '__main__':
    it_list_train = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    [x, w, y_ant, y_rffe, pwr_out] = parse_multiple_files(it_list_train)
    it_list_test = [16,17,18,19]
    [x_test, w_test, y_ant_test, y_rffe_test, pwr_out_test] = parse_multiple_files(it_list_test)
    # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    #Now its time for further training on specific models that has high success
    #load the results
    df = pd.read_csv(model_snr_recordings_file)
    for isnr in range(nsnr - 1, -1, -1):
        best_snr_performance=df['snr_{}'.format(isnr)].argmax()
        best_modelname = df['ModelName'][best_snr_performance]
        model_path = './model_checkpoints/' + best_modelname
        snr_model = tf.keras.models.load_model(model_path)
        nearby_snrs = [_ for _ in range(max(0,isnr-2),min(nsnr-1,isnr+2)+1)]
        record_folder = os.path.join( os.getcwd(),'optimized_models', 'snr_{}'.format(isnr))
        os.makedirs(record_folder,exist_ok=True)


        record_txt = os.path.join(record_folder, 'step_recording.txt')

        with open(record_txt, 'w') as f:
            f.write('ModelName,')
            for lsnr in range(nsnr - 1):
                f.write('snr_{},'.format(lsnr))
            f.write('snr_{}\n'.format(nsnr - 1))

        pred_snr = test(snr_model, y_rffe_test, pwr_out_test, x_test, w_test, 1, isnr)
        with open(record_txt, 'a') as f:
            f.write('{},'.format(best_modelname))
            for ksnr in range(nsnr - 1):
                f.write('{},'.format(pred_snr[ksnr]))
            f.write('{}\n'.format(pred_snr[nsnr - 1]))

        for ii in range(5):
            for jsnr in nearby_snrs:
                iteration_string = 'Training for target snr {}, iteration {}, neighbor {}'.format(isnr, ii, jsnr)
                print(iteration_string)
                r = np.hstack((y_ant[:, jsnr, :].real, y_ant[:, jsnr, :].imag))
                X = np.hstack((y_rffe[:, jsnr, :].real, y_rffe[:, jsnr, :].imag))
                r = my_norm(r)
                X = my_norm(X)
                X = np.hstack((X, 10 ** (0.1 * (pwr_out[:, jsnr, :] - 30))))

                r_test = np.hstack((y_ant_test[:, jsnr, :].real, y_ant_test[:, jsnr, :].imag))
                X_test = np.hstack((y_rffe_test[:, jsnr, :].real, y_rffe_test[:, jsnr, :].imag))
                r_test = my_norm(r_test)
                X_test = my_norm(X_test)
                X_test = np.hstack((X_test, 10 ** (0.1 * (pwr_out_test[:, jsnr, :] - 30))))

                snr_model.fit(X, r, epochs=10, batch_size=128, shuffle=True, verbose=0)

                model_name = best_modelname + '_OptimizedTarget_{}_{}_{}'.format(isnr,ii,jsnr)

                model_path = os.path.join(record_folder, model_name)
                snr_model.save(model_path)
                pred_snr = test(snr_model, y_rffe_test, pwr_out_test, x_test, w_test, ii, isnr)
                with open(record_txt, 'a') as f:
                    f.write('{},'.format(model_name))
                    for ksnr in range(nsnr - 1):
                        f.write('{},'.format(pred_snr[ksnr]))
                    f.write('{}\n'.format(pred_snr[nsnr - 1]))


