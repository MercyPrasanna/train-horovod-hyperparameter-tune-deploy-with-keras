# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import argparse
import os
import glob

import matplotlib.pyplot as plt
import tensorflow as tf
import horovod.tensorflow.keras as hvd

from azureml.core import Run
from utils import load_data, one_hot_encode

print("Keras version:", tf.keras.__version__)


# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--batch-size', type=int, dest='batch_size', default=50, help='mini batch size for training')
parser.add_argument('--first-layer-neurons', type=int, dest='n_hidden_1', default=100,
                    help='# of neurons in the first layer')
parser.add_argument('--second-layer-neurons', type=int, dest='n_hidden_2', default=100,
                    help='# of neurons in the second layer')
parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.001, help='learning rate')

args = parser.parse_args()

data_folder = args.data_folder

print('training dataset is stored here:', data_folder)

X_train_path = glob.glob(os.path.join(data_folder, '**/train-images-idx3-ubyte.gz'), recursive=True)[0]
X_test_path = glob.glob(os.path.join(data_folder, '**/t10k-images-idx3-ubyte.gz'), recursive=True)[0]
y_train_path = glob.glob(os.path.join(data_folder, '**/train-labels-idx1-ubyte.gz'), recursive=True)[0]
y_test_path = glob.glob(os.path.join(data_folder, '**/t10k-labels-idx1-ubyte.gz'), recursive=True)[0]

X_train = load_data(X_train_path, False) / 255.0
X_test = load_data(X_test_path, False) / 255.0
y_train = load_data(y_train_path, True).reshape(-1)
y_test = load_data(y_test_path, True).reshape(-1)

training_set_size = X_train.shape[0]

n_inputs = 28 * 28
n_h1 = args.n_hidden_1
n_h2 = args.n_hidden_2
n_outputs = 10
n_epochs = 20
batch_size = args.batch_size
learning_rate = args.learning_rate

y_train = one_hot_encode(y_train, n_outputs)
y_test = one_hot_encode(y_test, n_outputs)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, sep='\n')

# Build a simple MLP model
model = tf.keras.models.Sequential()
# first hidden layer
model.add(tf.keras.layers.Dense(n_h1, activation='relu', input_shape=(n_inputs,)))
# second hidden layer
model.add(tf.keras.layers.Dense(n_h2, activation='relu'))
# output layer
model.add(tf.keras.layers.Dense(n_outputs, activation='softmax'))

model.summary()


# Horovod: add Horovod DistributedOptimizer.
optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate*hvd.size())
optimizer = hvd.DistributedOptimizer(optimizer)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'], experimental_run_tf_function=False)

# start an Azure ML run
run = Run.get_context()


class LogRunMetrics(tf.keras.callbacks.Callback):
    # callback at the end of every epoch
    def on_epoch_end(self, epoch, log):
        # log a value repeated which creates a list
        run.log('Loss', log['loss'])
        run.log('Accuracy', log['acc'])


callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3, verbose=1)
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))
    callbacks.append(LogRunMetrics())
    callbacks.append(tf.keras.callbacks.TensorBoard(update_freq='batch'))

# Horovod: write logs on worker 0.
verbose = 1 if hvd.rank() == 0 else 0

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=n_epochs,
                    verbose=verbose,
                    validation_data=(X_test, y_test),
                    callbacks=callbacks)

score = model.evaluate(X_test, y_test, verbose=0)

# Horovod: write logs on worker 0.
verbose = 1 if hvd.rank() == 0 else 0

if hvd.rank() == 0:
    # log a single value
    run.log("Final test loss", score[0])
    print('Test loss:', score[0])

    run.log('Final test accuracy', score[1])
    print('Test accuracy:', score[1])

    plt.figure(figsize=(6, 3))
    plt.title('MNIST with Keras MLP ({} epochs)'.format(n_epochs), fontsize=14)
    plt.plot(history.history['acc'], 'b-', label='Accuracy', lw=4, alpha=0.5)
    plt.plot(history.history['loss'], 'r--', label='Loss', lw=4, alpha=0.5)
    plt.legend(fontsize=12)
    plt.grid(True)

    # log an image
    run.log_image('Accuracy vs Loss', plot=plt)

    # create a ./outputs/model folder in the compute target
    # files saved in the "./outputs" folder are automatically uploaded into run history
    os.makedirs('./outputs/model', exist_ok=True)

    # serialize NN architecture to JSON
    model_json = model.to_json()
    # save model JSON
    with open('./outputs/model/model.json', 'w') as f:
        f.write(model_json)
    # save model weights
    model.save_weights('./outputs/model/model.h5')
    print("model saved in ./outputs/model folder")
