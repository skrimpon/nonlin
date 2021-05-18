import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd


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
    uu=[parse_file(_) for _ in it_list]
    x       = np.concatenate([i[0] for i in uu],axis=0)
    w       = np.concatenate([i[1] for i in uu],axis=1)
    y_ant   = np.concatenate([i[2] for i in uu],axis=0)
    y_rffe  = np.concatenate([i[3] for i in uu], axis=0)
    pwr_out  = np.concatenate([i[4] for i in uu], axis=0)
    print('Testing on {} samples'.format(y_rffe.shape[0]))
    return [x, w, y_ant, y_rffe, pwr_out]


def snr(Phi, x, w):
    xh = sum(Phi.T * np.conj(w), 0) / np.sum(np.abs(w) ** 2, 0)
    a = np.mean(np.conj(xh) * x) / np.mean(np.abs(x) ** 2)
    d_var = np.mean(np.abs(xh - a * x) ** 2)
    snr_out = 10 * np.log10(np.abs(a) ** 2 / d_var)
    return snr_out


def test(y_rffe, pwr_out, x, w, models_folder):
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
        model_path = os.path.join(models_folder,'snr_{}'.format(isnr))
        snr_model = tf.keras.models.load_model(model_path)

        pred = snr_model(X).numpy()

        pred *= (X_max - X_min)
        pred += X_min

        pred = pred[:, :16] + 1j * pred[:, 16:]
        pred_snr[isnr] = snr(pred, x, w)
        base_snr[isnr] = snr(y_rffe[:, isnr, :], x, w)

    plt.figure()
    plt.title('Comparison of Baseline Method with our DNN approach')
    plt.plot(pin, base_snr, 'bs')
    plt.plot(pin, pred_snr, 'rd')
    plt.grid()
    plt.xlabel('Receive power per antenna [dBm]')
    plt.ylabel('Output SNR $\;(\gamma_\mathrm{out})\;$ [dB]')
    plt.legend(['Reference', 'DNN'])
    plt.show()
    return pred_snr,base_snr


if __name__ == '__main__':
    #Load test data
    it_list_test = [16,17,18,19]
    [x_test, w_test, y_ant_test, y_rffe_test, pwr_out_test] = parse_multiple_files(it_list_test)
    models_folder = os.path.join(os.getcwd(), 'optimized_models', 'best_models')
    pred_snr,base_snr= test(y_rffe_test, pwr_out_test, x_test, w_test, models_folder)


