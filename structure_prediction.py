import os
# suppress silly log messages from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import random
import structure_prediction_utils as utils
from tensorflow import keras
from keras.layers import Dropout, Conv2DTranspose, Conv2D, Dropout, UpSampling2D
from keras.applications import ResNet50
BATCH_SIZE=32

class ProteinStructurePredictor1(keras.Model):
    def __init__(self):
        super().__init__()
        self.resnet_base = ResNet50(weights=None, include_top=False, input_shape=(256, 256, 1))

        self.upsample1 = UpSampling2D(size=(2, 2))
        self.conv1 = Conv2D(128, kernel_size=3, padding='same', activation='relu')  
        self.upsample2 = UpSampling2D(size=(2, 2))  
        self.conv2 = Conv2D(64, kernel_size=3, padding='same', activation='relu')  
        self.upsample3 = UpSampling2D(size=(8, 8))
        self.dropout = Dropout(0.3)
        self.output_conv = Conv2D(1, kernel_size=3, padding='same', activation='linear')  

    def call(self, inputs, mask=None):
        distances_bc = self.prepare_input(inputs['primary_onehot'])

        x = self.resnet_base(distances_bc) 

        x = self.dropout(x)  
        x = self.upsample1(x) 
        x = self.conv1(x)    
        x = self.upsample2(x) 
        x = self.conv2(x) 
        x = self.upsample3(x) 
        
        output = self.output_conv(x)

        return tf.squeeze(output, axis=-1) 

    def prepare_input(self, primary_onehot):
        r = tf.range(0, utils.NUM_RESIDUES, dtype=tf.float32)
        distances = tf.abs(tf.expand_dims(r, -1) - tf.expand_dims(r, -2))
        distances_bc = tf.expand_dims(
            tf.broadcast_to(distances, [primary_onehot.shape[0], utils.NUM_RESIDUES, utils.NUM_RESIDUES]), -1)
        return tf.expand_dims(distances_bc, axis=-1)

class ProteinStructurePredictor0(keras.Model):
    def __init__(self):
        super().__init__()

        self.resnet_base = ResNet50(weights=None, include_top=False, input_shape=(256, 256, 1))
        self.conv_transpose1 = Conv2DTranspose(128, kernel_size=3, strides=(2, 2), padding='same', activation='relu')
        self.conv_transpose2 = Conv2DTranspose(64, kernel_size=3, strides=(2, 2), padding='same', activation='relu')  
        self.conv_transpose3 = Conv2DTranspose(32, kernel_size=3, strides=(8, 8), padding='same', activation='relu')  
        self.output_conv = Conv2D(1, kernel_size=3, padding='same', activation='linear')  

    def call(self, inputs, mask=None):
        distances_bc = self.prepare_input(inputs['primary_onehot'])

        x = self.resnet_base(distances_bc)  
        x = self.conv_transpose1(x)  
        x = self.conv_transpose2(x)  
        x = self.conv_transpose3(x) 
        output = self.output_conv(x) 

        return tf.squeeze(output, axis=-1)  

    def prepare_input(self, primary_onehot):
        r = tf.range(0, utils.NUM_RESIDUES, dtype=tf.float32)
        distances = tf.abs(tf.expand_dims(r, -1) - tf.expand_dims(r, -2))
        distances_bc = tf.expand_dims(
            tf.broadcast_to(distances, [primary_onehot.shape[0], utils.NUM_RESIDUES, utils.NUM_RESIDUES]), -1)
        return tf.expand_dims(distances_bc, axis=-1)  
    
def get_n_records(batch):
    return batch['primary_onehot'].shape[0]
def get_input_output_masks(batch):
    inputs = {'primary_onehot':batch['primary_onehot']}
    outputs = batch['true_distances']
    masks = batch['distance_mask']

    return inputs, outputs, masks
def train(model, train_dataset, validate_dataset=None, train_loss=utils.mse_loss):
    '''
    Trains the model
    '''

    avg_loss = 0.
    avg_mse_loss = 0.
    def print_loss():
        
        if validate_dataset is not None:
            validate_loss = 0.

            validate_batches = 0.
            for batch in validate_dataset.batch(BATCH_SIZE):
                validate_inputs, validate_outputs, validate_masks = get_input_output_masks(batch)
                validate_preds = model.call(validate_inputs, validate_masks)

                validate_loss += tf.reduce_sum(utils.mse_loss(validate_preds, validate_outputs, validate_masks)) / get_n_records(batch)
                validate_batches += 1
            validate_loss /= validate_batches
        else:
            validate_loss = float('NaN')
        print(
            f'train loss {avg_loss:.3f} train mse loss {avg_mse_loss:.3f} validate mse loss {validate_loss:.3f}')
    first = True
    for batch in train_dataset:
        inputs, labels, masks = get_input_output_masks(batch)
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            outputs = model(inputs, masks)

            l = train_loss(outputs, labels, masks)
            batch_loss = tf.reduce_sum(l)
            gradients = tape.gradient(batch_loss, model.trainable_weights)
            avg_loss = batch_loss / get_n_records(batch)
            avg_mse_loss = tf.reduce_sum(utils.mse_loss(outputs, labels, masks)) / get_n_records(batch)

        model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        print_loss()

        if first:
            print(model.summary())
            first = False

def test(model, test_records, viz=False):
    for batch in test_records.batch(BATCH_SIZE):
        test_inputs, test_outputs, test_masks = get_input_output_masks(batch)
        test_preds = model.call(test_inputs, test_masks)
        test_loss = tf.reduce_sum(utils.mse_loss(test_preds, test_outputs, test_masks)) / get_n_records(batch)
        print(f'test mse loss {test_loss:.3f}')

    if viz:
        print(model.summary())
        r = random.randint(0, test_preds.shape[0])
        utils.display_two_structures(test_preds[r], test_outputs[r], test_masks[r])

def main(data_folder):
    training_records = utils.load_preprocessed_data(data_folder, 'training.tfr')
    validate_records = utils.load_preprocessed_data(data_folder, 'validation.tfr')
    test_records = utils.load_preprocessed_data(data_folder, 'testing.tfr')

    model = ProteinStructurePredictor0()
    model.optimizer = keras.optimizers.Adam(learning_rate=1e-6)
    model.batch_size = BATCH_SIZE
    epochs = 5
    # Iterate over epochs.
    for epoch in range(epochs):
        epoch_training_records = training_records.shuffle(buffer_size=256).batch(model.batch_size, drop_remainder=False)
        print("Start of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        train(model, epoch_training_records, validate_records)

        test(model, test_records, True)

    test(model, test_records, True)

    model.save(os.path.join(data_folder, '/model'))


if __name__ == '__main__':
    local_home = os.path.expanduser("~")  # on my system this is /Users/jat171
    data_folder = r"data"

    main(data_folder)