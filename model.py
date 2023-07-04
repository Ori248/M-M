import tensorflow as tf
from constants import MEL_INPUT_SHAPE, QUADRANTS_TO_MOODS, N_SPLITS
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
from joblib import load, dump
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle


class Model:
    """
    The Music Mood Classifier using Convolutional Neural Network with TensorFlow Keras API
    attrs: input_shape: the input shape for the model
    """

    input_shape = MEL_INPUT_SHAPE

    def create_model(self):
        """
        The model initializer. It is not a __init__ constructor because it is used multiple times in the
        cross-validation process, and it is redundant to initialize it as soon as the class instance is created.
        """

        model = tf.keras.models.Sequential(name="Model")

        model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu',
                                         input_shape=self.input_shape, padding='same', name='Conv2D_1'))

        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 3), strides=(2, 2), padding='same',
                                               name='MaxPooling2D_1'))

        model.add(tf.keras.layers.BatchNormalization(name='BatchNorm1'))

        model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu',
                                         padding='same', name='Conv2D_2'))

        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 3), strides=(2, 2), padding='same',
                                               name='MaxPooling2D_2'))

        model.add(tf.keras.layers.BatchNormalization(name='BatchNorm2'))

        model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu',
                                         padding='same', name='Conv2D_3'))

        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 3), strides=(2, 2), padding='same',
                                               name='MaxPooling2D_3'))

        model.add(tf.keras.layers.BatchNormalization(name='BatchNorm3'))

        model.add(tf.keras.layers.Flatten(name='Flatten'))

        model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.2),
                                        name='Dense1'))

        model.add(tf.keras.layers.Dropout(rate=0.2, name='Dropout1'))

        model.add(tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.2),
                                        name='Dense2'))

        model.add(tf.keras.layers.Dropout(rate=0.2, name='Dropout2'))

        model.add(tf.keras.layers.Dense(4, activation='softmax', name='Softmax'))

        model.summary()

        self.draw_model(model)

        return model

    @staticmethod
    def draw_model(model):
        out_file = 'model.png'

        tf.keras.utils.plot_model(model, to_file=out_file, show_shapes=True)

    @staticmethod
    def shuffle(data, labels):
        """
        Shuffles the data and labels according to a randomly generated seed

        :param: data: the data to be shuffled
        :param: labels: the labels to be shuffled
        :return: shuffled data and labels
        """

        seed = random.randint(1, 100)

        print(f'Shuffling data and labels using seed {seed}...')

        data_tensor = tf.convert_to_tensor(data)

        labels_tensor = tf.convert_to_tensor(labels)

        shuffled_data = tf.random.shuffle(data_tensor, seed=seed)

        shuffled_labels = tf.random.shuffle(labels_tensor, seed=seed)

        return shuffled_data.numpy(), shuffled_labels.numpy()

    def train(self, data_path: Path, labels_path: Path, batch_size=32, num_epochs=200, learning_rate=1e-3):
        """
        Trains the dataset and tests it using KFold cross-validation. Saves models, training histories and
        confusion matrices for each fold.

        :param: data_path: path to the data file
        :param: labels_path: path to the labels file
        :default_params: hyperparameters for training
        :returns: None
        """

        kf = KFold(n_splits=N_SPLITS)

        data, labels = self.load_data_and_labels(data_path, labels_path)

        print(f'Starting audio model training with shapes: DATA {data.shape}, LABELS {labels.shape}')

        for i, (train, test) in enumerate(kf.split(data, labels)):
            print(f'Started fold {i}')

            model = self.create_model()

            loss_function = tf.keras.losses.SparseCategoricalCrossentropy()

            optimizer = tf.keras.optimizers.Adam()

            model.compile(optimizer=optimizer, loss=loss_function, metrics=['acc'])

            print('Created and compiled model. Starting to fit...')

            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

            checkpoint = tf.keras.callbacks.ModelCheckpoint(f'model{i}.h5', monitor='val_acc', mode='max',
                                                            verbose=1, save_best_only=True)

            history = model.fit(data[train], labels[train], batch_size=batch_size, epochs=num_epochs,
                                validation_data=(data[test], labels[test]), callbacks=[early_stopping, checkpoint])

            test_predictions = np.argmax(model.predict(data[test]), axis=1)

            conf_matrix = confusion_matrix(labels[test], test_predictions, normalize='pred')

            with open(f'history{i}.pkl', 'wb') as history_file:
                pickle.dump(history.history, history_file)

            dump(conf_matrix, f'confusion_matrix{i}.gz', compress=True)

            model.save(f'model{i}.h5')

    def load_data_and_labels(self, data_path, labels_path):
        """
        Loads the data and labels and returns a shuffled version of them

        :param: data_path: the data_path to be shuffled
        :param: labels_path: the labels_path to be shuffled
        :return: shuffled data and labels
        """

        data = load(data_path)

        labels = load(labels_path)

        return self.shuffle(data, labels)

    @staticmethod
    def visualize_history(history: dict, fold: int):
        """
        Visualizes training history of model.

        :param: history: the history to be visualized
        :param: fold: the fold number
        """
        plt.plot(history['acc'])

        plt.plot(history['val_acc'])

        plt.title(f'Model Accuracy - Fold {fold}')

        plt.ylabel('accuracy')

        plt.xlabel('epoch')

        plt.legend(['train', 'test'], loc='upper left')

        plt.show()

    @staticmethod
    def visualize_confusion_matrix(conf_matrix: np.ndarray, fold: int):
        """
        Visualizes confusion matrix of model.

        :param: conf_matrix: the confusion matrix to be visualized
        :param: fold: the fold number
        """
        display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                                         display_labels=np.array(list(QUADRANTS_TO_MOODS.keys())))

        display.plot()

        plt.title(f'Confusion Matrix - Fold {fold}')

        plt.show()

    def visualize_histories_and_confusion_matrices(self, dir_path: Path):
        """
        Iterates through history and confusion matrix files and visualizes them.
        :param: dir_path: path to the directory containing history and confusion matrix files
        """
        for i in range(N_SPLITS):
            with open(dir_path / f'history{i}.pkl', 'rb') as history_file:
                history = pickle.load(history_file)

            conf_matrix = load(dir_path / f'confusion_matrix{i}.gz')

            self.visualize_history(history, i)

            self.visualize_confusion_matrix(conf_matrix, i)
