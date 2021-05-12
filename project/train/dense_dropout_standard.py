import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, Normalizer, PowerTransformer
from sklearn.model_selection import train_test_split
from tensorflow.keras import losses
import time


# def snr(Phi, x):
#     xh = sum(Phi.T * np.conj(w), 0) / np.sum(np.abs(w) ** 2, 0)
#     a = np.mean(np.conj(xh) * x) / np.mean(np.abs(x) ** 2)
#     d_var = np.mean(np.abs(xh - a * x) ** 2)
#     snr_out = 10 * np.log10(np.abs(a) ** 2 / d_var)
#     return snr_out
#

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

#
# def make_model(input_shape):
#     model = keras.Sequential()
#     # Input
#     model.add(keras.Input(shape=(input_shape,)))
#     model.add(keras.layers.BatchNormalization())
#     # Dense 1
#     model.add(keras.layers.Dense(2048))
#     model.add(keras.layers.LeakyReLU())
#     # model.add(keras.layers.ReLU())
#     model.add(keras.layers.Dropout(0.3))
#     model.add(keras.layers.BatchNormalization())
#     # Dense 2
#     model.add(keras.layers.Dense(1024))
#     model.add(keras.layers.LeakyReLU())
#     # model.add(keras.layers.ReLU())
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

def make_model(input_shape):
    model = keras.Sequential()
    # Input
    model.add(keras.Input(shape=(input_shape,)))
    model.add(keras.layers.BatchNormalization())
    # Dense 1
    model.add(keras.layers.Dense(2048))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.BatchNormalization())
    # Dense 2
    model.add(keras.layers.Dense(1024))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.BatchNormalization())
    # Dense 3
    model.add(keras.layers.Dense(512))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.LeakyReLU())
    # Dense 4
    model.add(keras.layers.Dense(256))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.LeakyReLU())
    # Dense 4
    model.add(keras.layers.Dense(2))
    return model


def get_dataset(y_rffe,w,x,power_in,snrIdx=30):
    snr_interest_idx = snrIdx
    y_phi = y_rffe[:, snr_interest_idx, :]
    channel = w.transpose()
    p_in = np.array([10 ** (0.1 * power_in[snr_interest_idx])])
    snr_input = np.repeat(p_in[:, np.newaxis], y_phi.shape[0], axis=0)
    input_vector = np.concatenate([y_phi.real, y_phi.imag, channel.real, channel.imag, snr_input], 1)
    output_vector = np.concatenate([x[:,np.newaxis].real, x[:,np.newaxis].imag], 1)
    return input_vector,output_vector

idScenario=2
y_rffe, y_ant, w, x, power_in = load_data(idScenario=idScenario, it=1)



model_input_shape = 65
model = make_model(model_input_shape)
model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.MeanSquaredError())
# model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.MeanSquaredError())
all_inputs = np.empty([31*10000,65])
all_outputs = np.empty([31*10000,2])
for snrIdx in range(31):
    input_vector,output_vector = get_dataset(y_rffe,w,x,power_in,snrIdx=snrIdx)
    all_inputs[snrIdx*10000:(snrIdx+1)*10000,:] = input_vector
    all_outputs[snrIdx*10000:(snrIdx+1)*10000,:] = output_vector

xTrain, xTest, yTrain, yTest = train_test_split(all_inputs, all_outputs, shuffle=True, test_size=0.1)
# model.fit(xTrain, yTrain, epochs=50, batch_size=64, shuffle=True, validation_data=(xTest, yTest))
# BUFFER_SIZE=10000
BATCH_SIZE=256
datasetTrain = tf.data.Dataset.from_tensor_slices((xTrain, yTrain)).shuffle(xTrain.shape[0]).batch(BATCH_SIZE).prefetch(1024)
num_epochs = 20
model.fit(datasetTrain, epochs=num_epochs, validation_data=(xTest, yTest))

model.save('./trained_models-design{}/Dense-2048-2-EPOCH-{}'.format(idScenario,num_epochs))
num_load_epochs=20
snr_results = []
for snrIdx in range(31):
    model = keras.models.load_model('./trained_models-design{}/Dense-2048-2-EPOCH-{}'.format(idScenario,num_load_epochs))
    input_vector,output_vector = get_dataset(y_rffe,w,x,power_in,snrIdx=snrIdx)
    xTrain, xTest, yTrain, yTest = train_test_split(input_vector, output_vector, shuffle=True, test_size=0.1)
    model.fit(xTrain, yTrain, epochs=20, batch_size=64, shuffle=True, validation_data=(xTest, yTest))
    pred = model(input_vector)
    x_pred = tf.complex(real=pred[:, 0], imag=pred[:, 1]).numpy()
    x = tf.complex(real=output_vector[:,0], imag=output_vector[:,1]).numpy()
    a = np.mean(np.conj(x_pred) * x) / np.mean(np.abs(x) ** 2)
    d_var = np.mean(np.abs(x_pred - a * x) ** 2)
    snr_out = 10 * np.log10(np.abs(a) ** 2 / d_var)
    print('snrIdx: {}, snr_out: {}'.format(snrIdx, snr_out))
    snr_results.append([snrIdx,snr_out])

snr_res_np = np.array(snr_results)
plt.plot(power_in[snr_res_np[:,0]],snr_res_np[:,1])

y_ant_norm = y_ant / np.linalg.norm(y_ant,axis=0)
y_rffe_norm = y_rffe / np.linalg.norm(y_rffe,axis=0)
def snr(Phi):
    xh = sum(Phi.T*np.conj(w),0)/np.sum(np.abs(w)**2,0)
    a = np.mean(np.conj(xh)*x)/np.mean(np.abs(x)**2)
    d_var = np.mean(np.abs(xh - a*x)**2)
    snr_out = 10*np.log10(np.abs(a)**2/d_var)
    return snr_out
# Baseline data
base = y_rffe_norm

# Baseline data
gold = y_ant_norm

base_snr = np.zeros(31)
gold_snr = np.zeros(31)

for isnr in range(31):
    base_snr[isnr] = snr(base[:, isnr, :])
    gold_snr[isnr] = snr(gold[:, isnr, :])

plt.plot(power_in, base_snr, 'bs')
plt.plot(power_in, gold_snr, 'gx')
plt.grid()
plt.xlabel('Receive power per antenna [dBm]')
plt.ylabel('Output SNR $\;(\gamma_\mathrm{out})\;$ [dB]')
plt.legend(['Reference', 'Genie'])
#
# model_input_shape = 65
# model = make_model(model_input_shape)
# model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.MeanSquaredError())
# snr_results=np.zeros([10,31])
# for turnover in range(10):
#     for snrIdx in range(31):
#         input_vector,output_vector = get_dataset(y_rffe,w,x,power_in,snrIdx=snrIdx)
#         xTrain, xTest, yTrain, yTest = train_test_split(input_vector, output_vector, shuffle=True, test_size=0.1)
#
#         model.fit(xTrain, yTrain, epochs=10, batch_size=64, shuffle=True, validation_data=(xTest, yTest))
#
#         pred = model(input_vector)
#         x_pred = tf.complex(real=pred[:, 0], imag=pred[:, 1]).numpy()
#         x = tf.complex(real=output_vector[:,0], imag=output_vector[:,1]).numpy()
#         a = np.mean(np.conj(x_pred) * x) / np.mean(np.abs(x) ** 2)
#         d_var = np.mean(np.abs(x_pred - a * x) ** 2)
#         snr_out = 10 * np.log10(np.abs(a) ** 2 / d_var)
#         print('snrIdx: {}, snr_out: {}'.format(snrIdx, snr_out))
#         snr_results[turnover,snrIdx] = snr_out
# yyy = np.array(snr_results).transpose().reshape([-1, 31])


#
# a = np.mean(np.conj(xh) * x) / np.mean(np.abs(x) ** 2)
# d_var = np.mean(np.abs(xh - a * x) ** 2)
# snr_out = 10 * np.log10(np.abs(a) ** 2 / d_var)

#
# model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.MeanSquaredError())
#
# xTrain, xTest, yTrain, yTest = train_test_split(input_vector, x, shuffle=True, test_size=0.1)
#
#
# model.fit(xTrain, yTrain,epochs=10, batch_size=32, shuffle=True, validation_data=(xTest, yTest))
#
# pred = model(input_vector).numpy()
#
#
