import os
import numpy as np
import tensorflow as tf
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

f_optimized_models = os.path.join(os.getcwd(),'optimized_models')
final_models_dir = os.path.join(f_optimized_models,'best_models')
os.makedirs(final_models_dir,exist_ok=True)
combined_text = os.path.join(f_optimized_models,'combined_steps.txt')
with open(combined_text,'w') as f:
    f.write('ModelName,')
    for lsnr in range(nsnr - 1):
        f.write('snr_{},'.format(lsnr))
    f.write('snr_{}\n'.format(nsnr - 1))

snr_folders= os.listdir(f_optimized_models)

for folder in snr_folders:
    if folder.startswith('snr'):
        text_path = os.path.join(f_optimized_models,folder,'step_recording.txt')
        with open(text_path) as f2:
            lines = f2.readlines()
        with open(combined_text,'a') as f:
            for line in lines[2:]:
                f.write(line)



if __name__ == '__main__':

    #load the results
    df = pd.read_csv(combined_text)
    for isnr in range(nsnr - 1, -1, -1):
        best_snr_performance=df['snr_{}'.format(isnr)].argmax()
        best_modelname = df['ModelName'][best_snr_performance]
        model_snr_folder = os.path.join(f_optimized_models,'snr_' + best_modelname.split('OptimizedTarget_')[1].split('_')[0])
        model_path = os.path.join(model_snr_folder,best_modelname)
        snr_model = tf.keras.models.load_model(model_path)
        final_modelname = 'snr_{}'.format(isnr)
        final_modelpath = os.path.join(final_models_dir,final_modelname)
        snr_model.save(final_modelpath)

