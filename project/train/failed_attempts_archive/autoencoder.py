import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
from sklearn.preprocessing import StandardScaler, Normalizer, PowerTransformer
from sklearn.model_selection import train_test_split
import time

def load_data(idScenario=1, it=1):
    # Design id
    #
    # * 0: Non-linear front-end with ADC quantization
    # * 1: Non-linear front-end without ADC quantization
    # * 2: Linear front-end with ADC quantization
    # * 3: Linear front-end without ADC quantization

    # id = 0 #Design ID
    # it = 1  # iteration
    nrx = 16  # num of receiver antennas
    nsnr = 31  # num of snr points
    nx = 10000  # num of tx samples

    df = pd.read_csv(r'../../datasets/rx_1/param_1_' + str(idScenario + 1) + '_' + str(it) + '.csv')
    power_in = df['Pin']

    df = pd.read_csv(r'../../datasets/rx_1/idata_' + str(it) + '.csv')

    # Random tx data
    x = np.char.replace(np.array(df['x'], dtype=str), 'i', 'j').astype(np.complex)

    # Channel w
    w = np.array([np.char.replace(np.array(df['w_' + str(i + 1)], dtype=str), 'i', 'j').astype(np.complex)
                  for i in range(nrx)], dtype=complex)

    y_ant = np.array(
        [np.char.replace(np.array(df['yant_' + str(isnr * nrx + irx + 1)], dtype=str), 'i', 'j').astype(np.complex)
         for isnr in range(nsnr) for irx in range(nrx)], dtype=complex).T.reshape(nx * nsnr, nrx)

    df = pd.read_csv(r'../../datasets/rx_1/odata_' + str(idScenario + 1) + '_' + str(it) + '.csv')
    y_rffe = np.array(
        [np.char.replace(np.array(df['yrffe_' + str(isnr * nrx + irx + 1)], dtype=str), 'i', 'j').astype(np.complex)
         for isnr in range(nsnr) for irx in range(nrx)]).T.reshape(nx * nsnr, nrx)

    # # Print the shape for some of the arrays
    # print(f'y_ant shape: {y_ant.shape}')
    # print(f'y_rffe shape: {y_rffe.shape}')
    # print('w shape:{}'.format(w.shape))

    # Baseline data
    y_rffe = y_rffe.reshape(nx, nsnr, nrx)

    # Baseline data
    y_ant = y_ant.reshape(nx, nsnr, nrx)
    # print(f'y_ant shape: {y_ant.shape}')
    # print(f'y_rffe shape: {y_rffe.shape}')

    return y_rffe, y_ant, w, x, power_in


def get_dataset(y_rffe,w,x,power_in,snrIdx=30):
    snr_interest_idx = snrIdx
    y_phi = y_rffe[:, snr_interest_idx, :]
    channel = w.transpose()
    p_in = np.array([10 ** (0.1 * power_in[snr_interest_idx])])
    snr_input = np.repeat(p_in[:, np.newaxis], y_phi.shape[0], axis=0)
    input_vector = np.concatenate([y_phi.real, y_phi.imag, channel.real, channel.imag, snr_input], 1)
    output_vector = np.concatenate([x[:,np.newaxis].real, x[:,np.newaxis].imag], 1)
    return input_vector,output_vector

idScenario=3
y_rffe, y_ant, w, x, power_in = load_data(idScenario=idScenario, it=1)
w = w.transpose()

#This Block is for only 1 snr value
idxSnr=25
y_out = np.sqrt(y_rffe[:,idxSnr,:].shape[0])*y_rffe[:,idxSnr,:]/np.linalg.norm(y_rffe[:,idxSnr,:],axis=0)
zz = np.abs(y_out)
plt.figure()
plt.hist(zz)
y_in = np.sqrt(y_ant[:,idxSnr,:].shape[0])*y_ant[:,idxSnr,:]/np.linalg.norm(y_ant[:,idxSnr,:],axis=0)#/(10**(0.1*power_in[idxSnr]))
xx=np.abs(y_in)
plt.hist(xx)

#This block uses entire dataset irrespective of snr values.
y_out = np.sqrt(y_rffe.shape[0])*y_rffe/np.linalg.norm(y_rffe,axis=0)
# Baseline data
base = y_out
zz = np.abs(y_out.reshape([-1,16]))
plt.figure()
plt.hist(zz)
y_in = np.sqrt(y_ant.shape[0])*y_ant/np.linalg.norm(y_ant,axis=0)#/(10**(0.1*power_in[idxSnr]))
# Baseline data
gold = y_in
xx=np.abs(y_in.reshape([-1,16]))
plt.hist(xx)
y_out = y_out.reshape([-1,16])
y_in = y_in.reshape([-1,16])



w_rpt = w.repeat(31,axis=0)

bf_out = y_out*w_rpt
ii_in = np.concatenate([y_out[:,np.newaxis,:].real, y_out[:,np.newaxis,:].imag,
                        w_rpt[:,np.newaxis,:].real, w_rpt[:,np.newaxis,:].imag,
                        bf_out[:,np.newaxis,:].real,bf_out[:,np.newaxis,:].imag],axis=1)

ii_in = ii_in[...,np.newaxis]
ii_out= np.concatenate([y_in[:,np.newaxis,:,np.newaxis].real, y_in[:,np.newaxis,:,np.newaxis].imag],axis=1)

class Invertor(Model):
  def __init__(self):
    super(Invertor, self).__init__()
    self.encoder = tf.keras.Sequential([
        layers.Input(shape=(6, 16, 1)),
        layers.Conv2D(16, (6, 16), activation=keras.layers.LeakyReLU(), padding='same', strides=3),
        layers.Conv2D(8, (2, 2), activation=keras.layers.LeakyReLU(), padding='same', strides=3)])

    self.decoder = tf.keras.Sequential([
        layers.Conv2DTranspose(2, kernel_size=3, strides=(1,2), activation='linear', padding='same'),
        layers.Conv2DTranspose(16, kernel_size=3, strides=(2,4), activation='linear', padding='same'),
        layers.Conv2D(1, kernel_size=(3, 3), activation='linear', padding='same')])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


model_inv = Invertor()
model_inv.compile(optimizer='adam', loss=losses.MeanSquaredError())
model_inv.fit(ii_in, ii_out, epochs=10, shuffle=True, batch_size=64)

denoised = model_inv(ii_in).numpy()[:,:,:,0]

y_denois = denoised[:,0,:] + 1j* denoised[:,1,:]
pred = y_denois.reshape([10000,31,16])

w=w.transpose()
def snr(Phi):
    xh = sum(Phi.T*np.conj(w),0)/np.sum(np.abs(w)**2,0)
    a = np.mean(np.conj(xh)*x)/np.mean(np.abs(x)**2)
    d_var = np.mean(np.abs(xh - a*x)**2)
    snr_out = 10*np.log10(np.abs(a)**2/d_var)
    return snr_out


nsnr=31
pred_snr = np.zeros(nsnr)
base_snr = np.zeros(nsnr)
gold_snr = np.zeros(nsnr)
for isnr in range(nsnr):
    pred_snr[isnr] = snr(pred[:, isnr, :])
    base_snr[isnr] = snr(base[:, isnr, :])
    gold_snr[isnr] = snr(gold[:, isnr, :])

plt.plot(power_in, base_snr, 'bs')
plt.plot(power_in, pred_snr, 'rd')
plt.plot(power_in, gold_snr, 'gx')
plt.grid()
plt.xlabel('Receive power per antenna [dBm]')
plt.ylabel('Output SNR $\;(\gamma_\mathrm{out})\;$ [dB]')
plt.legend(['Reference', 'DNN', 'Genie'])


#
#
# x_pred = tf.complex(real=pred[:, 0], imag=pred[:, 1]).numpy()
# x = tf.complex(real=output_vector[:, 0], imag=output_vector[:, 1]).numpy()
# a = np.mean(np.conj(x_pred) * x) / np.mean(np.abs(x) ** 2)
# d_var = np.mean(np.abs(x_pred - a * x) ** 2)
# snr_out = 10 * np.log10(np.abs(a) ** 2 / d_var)
# print('snrIdx: {}, snr_out: {}'.format(snrIdx, snr_out))