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


##=================================================================
## MODEL
##================================================================
def model_loss(y_true, y_pred):
    return tf.keras.losses.MSE(y_true, y_pred)


model_optimizer = tf.keras.optimizers.Adam(1e-3)


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(input,output):
    # real = tf.constant([[1.0], [0.0]])
    # imag = tf.constant([[0.0], [1.0]])
    # complex_converter = tf.complex(real, imag)
    with tf.GradientTape() as model_tape:
        yPred = model(input, training=True)
        # y_pred = tf.complex(real=yPred[:, 0], imag=yPred[:, 1])
        # y_pred = tf.matmul(tf.dtypes.cast(yPred, tf.complex64),complex_converter)
        mod_loss = tf.keras.losses.MSE(output, yPred)
        model_gradient = model_tape.gradient(mod_loss, model.trainable_variables)
        model_optimizer.apply_gradients(zip(model_gradient, model.trainable_variables))


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        for batch in dataset:
            input_batch = batch[0]
            output_batch = batch[1]
            train_step(input_batch,output_batch)
        # # Save the model every 15 epochs
        # if (epoch + 1) % 15 == 0:
        #     checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))


def make_model(input_shape):
    model = keras.Sequential()
    # Input
    model.add(keras.Input(shape=(input_shape,)))
    model.add(keras.layers.BatchNormalization())
    # Dense 1
    model.add(keras.layers.Dense(2048))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.BatchNormalization())
    # Dense 2
    model.add(keras.layers.Dense(1024))
    model.add(keras.layers.LeakyReLU())
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


def get_dataset(y_rffe,y_ant,w,x,power_in,snrIdx=30):
    # y_rffe, y_ant, w, x, power_in = load_data(idScenario=idScenario, it=it)
    snr_interest_idx = snrIdx
    y_phi = y_rffe[:, snr_interest_idx, :]
    yp_max = np.max(np.abs(y_phi))
    # y_phi = y_phi /yp_max
    # print('mean:{}'.format(np.mean(y_phi,axis=0)))
    # print('variance:{}'.format(np.var(y_phi,axis=0)))
    channel = w.transpose()
    y_orig = y_ant[:, snr_interest_idx, :]
    y_denoise = np.transpose(w * x)
    p_in = np.array([10 ** (0.1 * power_in[snr_interest_idx])])
    snr_input = np.repeat(p_in[:, np.newaxis], y_phi.shape[0], axis=0)
    input_vector = np.concatenate([y_phi.real, y_phi.imag, channel.real, channel.imag, snr_input], 1)
    output_vector = np.concatenate([x[:,np.newaxis].real, x[:,np.newaxis].imag], 1)
    # print('Created an input where each element is combination of')
    # print('y_rffe.real, y_rffe.imag, channel.real, channel.imag, power_in_linear')
    # print('Input has shape : {}'.format(input_vector.shape))

    input_shape = input_vector.shape[1]
    BUFFER_SIZE = input_vector.shape[0]
    BATCH_SIZE = 32
    datasetTrain = tf.data.Dataset.from_tensor_slices((input_vector, output_vector)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return datasetTrain
    # xTrain, xTest, yTrain, yTest = train_test_split(input_vector, x, shuffle=True, test_size=0.1)
    # BUFFER_SIZE = xTrain.shape[0]
    # BATCH_SIZE = 32
    # datasetTrain = tf.data.Dataset.from_tensor_slices((xTrain, yTrain)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    # datasetTest = tf.data.Dataset.from_tensor_slices((xTest, yTest)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# model = make_model(65)
# datasetTrain = get_dataset(idScenario=1,it=1,snrIdx=30)
#
# train(datasetTrain, 50)
#
# pred = model(input_data).numpy()
# x_pred = tf.complex(real=pred[:, 0], imag=pred[:, 1])
# a = np.mean(np.conj(x_pred) * x_new) / np.mean(np.abs(x_new) ** 2)
# d_var = np.mean(np.abs(x_pred - a * x_new) ** 2)
# snr_out = 10 * np.log10(np.abs(a) ** 2 / d_var)
# print('snr_out: {}'.format(snr_out))
#
# pred = model(input_vector).numpy()
# x_pred = tf.complex(real=pred[:, 0], imag=pred[:, 1])
# a = np.mean(np.conj(x_pred) * x) / np.mean(np.abs(x) ** 2)
# d_var = np.mean(np.abs(x_pred - a * x) ** 2)
# snr_out = 10 * np.log10(np.abs(a) ** 2 / d_var)
# print('snr_out: {}'.format(snr_out))


#
#
# dataset = get_dataset(idScenario=1, it=1,snrIdx=30)
# for iter in [2,3,4,5]:
#     dataset.concatenate(get_dataset(idScenario=1, it=iter))
model_input_shape = 65#dataset.element_spec[0].shape[1]
model = make_model(model_input_shape)

idScenario=1
it=1
y_rffe, y_ant, w, x, power_in = load_data(idScenario=idScenario, it=it)
for epoch in range(10):
    # print('EPOCH : {}'.format(epoch))
    for snrIdx in range(31):
        dataset = get_dataset(y_rffe,y_ant,w,x,power_in,snrIdx=snrIdx)
        train(dataset, 10)
        input_data = []
        x_data = []
        for data in dataset:
            input_data.append(data[0])
            x_data.append(data[1])
        input_data = tf.concat(input_data,axis=0).numpy()
        x = tf.concat(x_data,axis=0).numpy()
        x = tf.complex(real=x[:, 0], imag=x[:, 1]).numpy()
        pred = model(input_data,training=False).numpy()
        x_pred = tf.complex(real=pred[:, 0], imag=pred[:, 1])
        a = np.mean(np.conj(x_pred) * x) / np.mean(np.abs(x) ** 2)
        d_var = np.mean(np.abs(x_pred - a * x) ** 2)
        snr_out = 10 * np.log10(np.abs(a) ** 2 / d_var)
        print('snr_out: {}'.format(snr_out))

# for iter in range(1,5):
#     dataset = get_dataset(idScenario=1,it=iter)
#     train(dataset, 50)
#     pred = model(input_vector).numpy()
#     x_pred = tf.complex(real=pred[:, 0], imag=pred[:, 1])
#     a = np.mean(np.conj(x_pred) * x) / np.mean(np.abs(x) ** 2)
#     d_var = np.mean(np.abs(x_pred - a * x) ** 2)
#     snr_out = 10 * np.log10(np.abs(a) ** 2 / d_var)
#     print('snr_out: {}'.format(snr_out))




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
# a = np.mean(np.conj(pred) * x) / np.mean(np.abs(x) ** 2)
# d_var = np.mean(np.abs(pred - a * x) ** 2)
# snr_out = 10 * np.log10(np.abs(a) ** 2 / d_var)


#
# def model_loss(y_true,y_pred):
#     return tf.keras.losses.MSE(y_true, y_pred)
#
# model_optimizer = tf.keras.optimizers.Adam(1e-3)
#
#
# # Notice the use of `tf.function`
# # This annotation causes the function to be "compiled".
# @tf.function
# def train_step(x_samples,y_samples):
#     with tf.GradientTape() as model_tape:
#         yPred = model(x_samples, training=True)
#         mod_loss = model_loss(y_samples,yPred)
#         model_gradient = model_tape.gradient(mod_loss,model.trainable_variables)
#         model_optimizer.apply_gradients(zip(model_gradient,model.trainable_variables))
#
#
# def train(xTrain,yTrain,epochs):
#     for epoch in range(epochs):
#         start = time.time()
#
#         for image_batch in dataset:
#             train_step(image_batch)
#
#         # Produce images for the GIF as you go
#         display.clear_output(wait=True)
#         generate_and_save_images(generator,
#                                  epoch + 1,
#                                  seed)
#
#         # Save the model every 15 epochs
#         if (epoch + 1) % 15 == 0:
#             checkpoint.save(file_prefix=checkpoint_prefix)
#
#         print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
#
#         # Generate after the final epoch
#     display.clear_output(wait=True)
#     generate_and_save_images(generator,
#                              epochs,
#                              seed)

# x_train, x_test, y_train, y_test = train_test_split(X, r, shuffle=True, test_size=0.1)
#
# # scale_in = StandardScaler(with_mean=True, with_std=True).fit(x_train)
# # scale_out = StandardScaler(with_mean=True, with_std=True).fit(y_train)
# scale_in = PowerTransformer().fit(x_train)
# scale_out = PowerTransformer().fit(y_train)
#
# x_train = scale_in.transform(x_train)
# x_test = scale_in.transform(x_test)
# y_train = scale_out.transform(y_train)
# y_test = scale_out.transform(y_test)
#
# print(f'x_train shape: {x_train.shape}')
#

# # Use the NN to predict the new data
# pred = model(scale_in.transform(X)).numpy()
#
# # Find the complex data
# pred = pred[:, :16] + 1j * pred[:, 16:]
# pred = pred.reshape(nx, nsnr, nrx)
#
# # Baseline data
# base = y_rffe.reshape(nx, nsnr, nrx)
#
# # Baseline data
# gold = y_ant.reshape(nx, nsnr, nrx)
#
# pred_snr = np.zeros(nsnr)
# base_snr = np.zeros(nsnr)
# gold_snr = np.zeros(nsnr)
#
# for isnr in range(nsnr):
#     pred_snr[isnr] = snr(pred[:, isnr, :])
#     base_snr[isnr] = snr(base[:, isnr, :])
# #     gold_snr[isnr] = snr(gold[:, isnr, :])
#
# plt.plot(power_in, base_snr, 'bs')
# plt.plot(power_in, pred_snr, 'rd')
# # plt.plot(power_in, gold_snr, 'gx')
# plt.grid()
# plt.xlabel('Receive power per antenna [dBm]')
# plt.ylabel('Output SNR $\;(\gamma_\mathrm{out})\;$ [dB]')
# plt.legend(['Reference', 'DNN'])
# # plt.legend(['Reference', 'DNN', 'Genie'])
