# This script contains all the essential functions for managing a Deep Learning project within TensorFlow framework
import datetime
import itertools
import os
import random

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Importing ML packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import zipfile


class Prediction:

    def __init__(self):
        pass

    # Create a function to import an image and resize it to be able to be used with the model
    @staticmethod
    def preprocess_image(filename, img_shape=224, scale=True):
        """
        Reads in an image from filename, turns it into a tensor and reshapes into
        (224, 224, 3).

        :param filename: string filename of target image
        :param img_shape: size to resize target image to default 224 x 224
        :param scale: Normalizes pixels to range(0, 1), default True
        :return: Returns the preprocessed image
        """

        # Read in the image
        img = tf.io.read_file(filename)
        # Decode it into tensor
        img = tf.image.decode_jpeg(img)
        # Resize the image
        img = tf.image.resize(img, [img_shape, img_shape])

        if scale:
            # Rescale the image (get all values between 0 and 1)
            return img / 255.
        else:
            return img

    # Make a function to predict on images and plot them (works with multi-class)
    def pred_plot(self, model, filename, class_names):
        """
        Imports an image located at filename, makes a prediction on it with
        a trained model and plots the image with the predicted class as the title.

        :param model: Trained deep neural network model
        :param filename: Custom input image file
        :param class_names: Unique Classes from the data
        :return: None
        """

        # Import the target image and preprocess it
        img = self.preprocess_image(filename)
        # Make a prediction
        pred = model.predict(tf.expand_dims(img, axis=0))

        # Get the predicted class
        if len(class_names) > 1:  # check for multi-class
            pred_class = class_names[pred.argmax()]
        else:
            pred_class = class_names[int(tf.round(pred)[0][0])]  # if only one output, round

        # Plot the image and predicted class
        plt.imshow(img)
        plt.title(f"Prediction: {pred_class}")
        plt.axis(False)
        plt.show()


class DataHandler:

    def __init__(self):
        pass

    @staticmethod
    def unzip_data(filename):
        """
        Unzips filename into the current working directory.

        :param filename: a filepath to a target zip folder to be unzipped.
        :return: data
        """

        zip_ref = zipfile.ZipFile(filename, 'r')
        zip_ref.extractall()
        zip_ref.close()

    @staticmethod
    def walk_through_dir(dir_path):
        """
        Walks through dir_path returning its contents.

        :param dir_path: Data Directory
        :return: A print out of:
                number of subdirectories in dir_path
                number of images (files) in each subdirectory
                name of each subdirectory
        """

        for dir_path, dir_names, filenames in os.walk(dir_path):
            print(f"There are {len(dir_names)} directories and {len(filenames)} images in '{dir_path}'.")


class Callback:

    def __init__(self):
        pass

    @staticmethod
    def model_checkpoint(checkpoint_path):
        """
        Creates a ModelCheckpoint callback that saves the model's weights only.

        :param checkpoint_path: the filepath to save the model's weights
        :return: tf.keras.callbacks.ModelCheckpoint()
        """

        return tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True,
                                                  monitor='val_accuracy', save_best_only=False,
                                                  verbose=1)

    @staticmethod
    def create_tensorboard(dir_name, experiment_name):
        """
        Creates a TensorBoard callback instance and stores log files.

        Stores log files with the filepath:
        "dir_name/experiment_name/current_datetime/"

        :param dir_name: target directory to store TensorBoard log files
        :param experiment_name: name of experiment directory (e.g. efficientnet_model_1)
        :return: TensorBoard
        """

        log_dir = dir_name + '/' + experiment_name + '/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

        print(f'Saving TensorBoard log files to: {log_dir}')

        return tensorboard_callback


class ModelEvaluation:

    def __init__(self):
        pass

    @staticmethod
    def results(y_true, y_pred):
        """
        Calculates model accuracy, precision, recall, and f1 score
        :param y_true: true labels in the form of a 1D array
        :param y_pred: predicted labels in the form of a 1D array
        :return: Dictionary of accuracy, precision, recall, f1-score
        """

        # Calculate model accuracy
        model_accuracy = accuracy_score(y_true, y_pred) * 100
        # Calculate model precision, recall, and f1 score using weighted average
        model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        model_results = {
            'accuracy': model_accuracy,
            'precision': model_precision,
            'recall': model_recall,
            'f1': model_f1
        }

        return model_results

    @staticmethod
    def compare_history(original_history, new_history, initial_epochs=5):
        """
        Compares two TensorFlow model History objects.

        :param original_history: History object from original model (before new_history)
        :param new_history: History object from continued model training (after original_history)
        :param initial_epochs: Number of epochs in original_history (new_history plot starts from here)
        :return: None
        """

        # Get original history measurements
        acc = original_history.history['accuracy']
        loss = original_history.history['loss']

        val_acc = original_history.history['val_accuracy']
        val_loss = original_history.history['val_loss']

        # Get the combined original and new history measurements
        total_acc = acc + new_history.history['accuracy']
        total_loss = loss + new_history.history['loss']

        total_val_acc = val_acc + new_history.history['val_accuracy']
        total_val_loss = val_loss + new_history.history['val_loss']

        # Make plots
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(total_acc, label='Training Accuracy')
        plt.plot(total_val_acc, label='Validation Accuracy')
        plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(),
                 label='Fine Tuning Point')  # re-shift plot around epochs
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(total_loss, label='Training Loss')
        plt.plot(total_val_loss, label='Validation Loss')
        plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(),
                 label='Fine Tuning Point')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.grid(True)

        plt.show()

    @staticmethod
    def loss_acc_curves(history):
        """
        Plots separate loss and accuracy curves for training and validation metrics.

        :param history: TensorFlow model History object
        (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
        :return: None
        """

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']

        epochs = range(len(loss))

        # create a plot
        plt.figure(figsize=(12, 6))

        # Plot loss
        plt.subplot(2, 1, 1)
        plt.plot(epochs, loss, label='training_loss')
        plt.plot(epochs, val_loss, label='val_loss')
        plt.title('Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Plot accuracy
        plt.subplot(2, 1, 2)
        plt.plot(epochs, accuracy, label='training_accuracy')
        plt.plot(epochs, val_accuracy, label='val_accuracy')
        plt.title('Accuracy Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    # Note: The following confusion matrix code is a remix of Scikit-Learn's plot_confusion_matrix function -
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
    @staticmethod
    def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10),
                              text_size=15, norm=False, savefig=False):
        """
        Makes a labelled confusion matrix will be labelled, if not, integer class values
        will be used.

        :param y_true: Array of truth labels (must be same shape as y_pred).
        :param y_pred: Array of predicted labels (must be same shape as y_true).
        :param classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
        :param figsize: Size of output figure (default=(10, 10)).
        :param text_size: Size of output figure text (default=15).
        :param norm: normalize values or not (default=False).
        :param savefig: save confusion matrix to file (default=False).
        :return: A labelled confusion matrix plot comparing y_true and y_pred.

        Example usage:
        make_confusion_matrix(y_true=test_labels, # ground truth test labels
                              y_pred=y_preds, # predicted labels
                              classes=class_names, # array of class label names
                              figsize=figsize(15, 15),
                              text_size=10)
        """

        # Create the confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # normalize it
        n_classes = cm.shape[0]  # find the number of classes we're dealing with

        # plot the figure and make it pretty
        fig, ax = plt.subplots(figsize=figsize)
        cax = ax.matshow(cm, cmap=plt.cm.Greens)
        fig.colorbar(cax)

        # Are there a list of classes?
        if classes:
            labels = classes
        else:
            labels = np.arange(cm.shape[0])

        # Label the axes
        ax.set(title='Confusion Matrix',
               xlabel='Predicted label', ylabel='True label',
               xticks=np.arange(n_classes),  # create enough axis slots for each class
               y_ticks=np.arange(n_classes),
               xticklabels=labels,  # axes will be labeled with class names (if they exist) or ints
               yticklabels=labels)

        # Make x-axis labels appear on bottom
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()

        # Set the threshold for different colors
        threshold = (cm.max() + cm.min()) / 2.

        # Plot the text on each cell
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

            if norm:
                plt.text(j, i, f'{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)',
                         horizontalalignment='center',
                         color='white' if cm[i, j] > threshold else 'black',
                         size=text_size)
            else:
                plt.text(j, i, f'{cm[i, j]}',
                         horizontalalignment='center',
                         color='white' if cm[i, j] > threshold else 'black',
                         size=text_size)

        # Save the figure to the current working directory
        if savefig:
            fig.savefig('confusion_matrix.png')


class Visualize:

    def __init__(self):
        pass

    @staticmethod
    def view_random_image(target_dir, target_class):
        """
        View random image from a target directory and class

        :param target_dir: target directory path
        :param target_class: target class folder
        :return: image
        """

        # Setup target directory (we'll view images from here)
        target_folder = target_dir + '/' + target_class

        # Get a random image path
        random_image = random.sample(os.listdir(target_folder), 1)

        # Read in the image and plot it using matplotlib
        img = mpimg.imread(target_folder + '/' + random_image[0])
        plt.imshow(img)
        plt.title(target_class)
        plt.axis('off')
        plt.show()

        print(f'Image shape: {img.shape}')  # show the shape of the image

        return img

    @staticmethod
    def plot_random_images(model, images, true_labels, classes, n=5):
        """
        Picks a random image, plots it and labels it with a prediction and ground truth.

        :param model: trained model that fit to the training data
        :param images: image tensors from the dataset
        :param true_labels: ground truth
        :param classes: collection of unique classes in the dataset
        :param n: no. of classes
        :return: None
        """

        # Plot the image
        plt.figure(figsize=(15, 10))
        for j in range(n):

            # Pick a random image
            i = random.randint(0, len(images))

            # Create predictions and targets
            target_image = images[i]

            # Create predictions and targets
            pred_probs = model.predict(target_image.reshape(1, 28, 28), verbose=0)
            pred_label = classes[pred_probs.argmax()]
            true_label = classes[true_labels[i]]

            # Plot the image
            plt.subplot(n // 5, 5, j + 1)
            plt.imshow(target_image, cmap=plt.cm.binary)

            # Change the color of the titles depending on if the prediction is right or wrong
            if pred_label == true_label:
                color = 'green'
            else:
                color = 'red'

            # Add xlabel information (prediction/true label)
            plt.xlabel('Pred: {} {:2.0f}% (True: {})'.format(pred_label,
                                                             100 * tf.reduce_max(pred_probs), true_label),
                       color=color)

        # Adjust layout to prevent overlapping
        plt.tight_layout()
        plt.show()

    @staticmethod
    def view_random_images(train_dir):
        """
        Plots random images from the dataset

        :param train_dir: training directory that contains sub-folders called classes
        :return: 3 x 3 grid images
        """

        classes = os.listdir(train_dir)

        # Creating a figure and subplot
        plt.figure(figsize=(15, 10))
        for i in range(16):

            # Pick random class
            target_class = random.sample(classes, 1)
            target_dir = train_dir + '/' + target_class[0]

            # Pick a random image
            image = random.sample(os.listdir(target_dir), 1)

            # Read in the image and plot it using matplotlib
            img = mpimg.imread(target_dir + '/' + image[0])

            # plot the img
            plt.subplot(4, 4, i+1)
            plt.imshow(img)
            plt.title(target_class[0])
            plt.axis('off')

        plt.tight_layout()
        plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Running...')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
