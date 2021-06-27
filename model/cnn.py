"""CNN Model"""
import os.path


from .base_model import BaseModel
from dataloader.dataloader import DataLoader

# Disabled debugging logs using os.environ
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

import numpy as np


class CNN(BaseModel):
    """CNN Model Class"""

    def __init__(self, config):
        super().__init__(config)

        self.model = None

        self.base_dir = None
        self.train_dir = None
        self.test_dir = None
        self.valid_dir = None

        self.train_data = None
        self.test_data = None
        self.valid_data = None

        self.batch_size = self.config.train.batch_size
        self.epochs = self.config.train.epochs

    def load_data(self):
        """Loading and Preprocessing data"""
        self.base_dir = DataLoader().load_data(self.config.data)
        self.train_dir = os.path.join(self.base_dir, 'train')
        self.test_dir = os.path.join(self.base_dir, 'test')
        self.valid_dir = os.path.join(self.base_dir, 'valid')

        train_datagen = ImageDataGenerator(rescale=1.0 / 255.)
        test_datagen = ImageDataGenerator(rescale=1.0 / 255.)
        valid_datagen = ImageDataGenerator(rescale=1.0 / 255.)

        train_generator = train_datagen.flow_from_directory(self.train_dir, batch_size=self.config.train.batch_size,
                                                            class_mode=self.config.data.class_mode,
                                                            target_size=(150, 150))

        test_generator = test_datagen.flow_from_directory(self.test_dir, batch_size=self.config.train.batch_size,
                                                          class_mode=self.config.data.class_mode,
                                                          target_size=(150, 150))

        valid_generator = valid_datagen.flow_from_directory(self.valid_dir, batch_size=self.config.train.batch_size,
                                                            class_mode=self.config.data.class_mode,
                                                            target_size=(150, 150))

        return train_generator, test_generator, valid_generator

    def build(self):
        """Build CNN base model"""
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv2D(self.config.model.layers_stack.layer_1,
                                       self.config.model.layers_stack.kernel_size, activation='relu',
                                       input_shape=(150, 150, 3)),
                tf.keras.layers.MaxPool2D(self.config.model.layers_stack.pool_size),

                tf.keras.layers.Conv2D(self.config.model.layers_stack.layer_2,
                                       self.config.model.layers_stack.kernel_size, activation='relu',
                                       input_shape=(150, 150, 3)),
                tf.keras.layers.MaxPool2D(self.config.model.layers_stack.pool_size),

                tf.keras.layers.Conv2D(self.config.model.layers_stack.layer_3,
                                       self.config.model.layers_stack.kernel_size, activation='relu',
                                       input_shape=(150, 150, 3)),
                tf.keras.layers.MaxPool2D(self.config.model.layers_stack.pool_size),

                tf.keras.layers.Flatten(),

                tf.keras.layers.Dense(1064, activation='relu'),
                tf.keras.layers.Dense(self.config.model.output, activation='softmax'),

            ]
        )

    def train(self):
        """Compile and train the model"""
        self.model.compile(optimizer=self.config.train.optimizer.type,
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=self.config.train.metrics)

        self.train_data, self.test_data, self.valid_data = self.load_data()

        model_history = self.model.fit(self.train_data, validation_data=self.valid_data,
                                       epochs=self.config.train.epochs, verbose=1)

        return model_history.history['loss'], model_history.history['val_loss']

    def evaluate(self):
        """ Evaluating on Test data"""
        print('\nEvaluating')
        test_loss, test_acc = self.model.evaluate(self.test_data)
        print("Loss ", test_loss)
        print("Test Accuracy", test_acc)

    def predicting(self):
        """ Making Prediction on Test data """
        print("\nMaking Prediction")
        test_image = image.load_img("E:/Machine Learning Series/Datasets/archive/real_vs_fake/real-vs-fake/test/fake"
                                    "/0EDS0OZ1XY.jpg", target_size=(150, 150, 3))

        class_name = ['fake', 'real']

        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = self.model.predict(test_image)

        print("This image {} with {:.2f} percentage. ".format(class_name[np.argmax(result)], 100 * np.max(result)))

    def saving(self):
        """ Saving the model"""
        print("\nSaving the model")
        self.model.save("E:/FYP 2021/Final Project/saved_models/model_1.h5")
        print(">>>> Model Saved - ALL DONE <<<<")
