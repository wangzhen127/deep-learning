import helper
import numpy as np
from os.path import isfile, isdir
import pickle
import problem_unittests as tests
import random
from sklearn import preprocessing
import tarfile
import time
import tensorflow as tf
from tqdm import tqdm
from urllib.request import urlretrieve

cifar10_dataset_folder_path = 'cifar-10-batches-py'
tar_gz_path = 'cifar-10-python.tar.gz'


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def DownloadDataSet():
    if not isfile(tar_gz_path):
        with DLProgress(unit='B',
                        unit_scale=True,
                        miniters=1,
                        desc='CIFAR-10 Dataset') as pbar:
            urlretrieve(
                'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                tar_gz_path,
                pbar.hook)

    if not isdir(cifar10_dataset_folder_path):
        with tarfile.open(tar_gz_path) as tar:
            tar.extractall()
            tar.close()

def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    range_low = 0
    range_high = 1
    min_value = 0
    max_value = 255
    return range_low + (
        ( (x - min_value)*(range_high - range_low) )/( max_value - min_value ) )


def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for
    each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    LABELS = np.array([0,1,2,3,4,5,6,7,8,9])
    lb = preprocessing.LabelBinarizer()
    lb.fit(LABELS)
    return lb.transform(x)


def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    # Apply 1, 2, or 3 Convolution and Max Pool layers
    conv = tf.contrib.layers.conv2d(
        inputs=x,
        num_outputs=64,
        kernel_size=3,
        stride=1,
        padding='SAME',
        activation_fn=tf.nn.relu,
        weights_initializer=tf.truncated_normal_initializer,
        biases_initializer=tf.zeros_initializer)
    conv = tf.contrib.layers.max_pool2d(
        inputs=conv,
        kernel_size=2,
        stride=2,
        padding='SAME')

    conv = tf.contrib.layers.conv2d(
        inputs=conv,
        num_outputs=128,
        kernel_size=3,
        stride=1,
        padding='SAME',
        activation_fn=tf.nn.relu,
        weights_initializer=tf.truncated_normal_initializer,
        biases_initializer=tf.zeros_initializer)
    conv = tf.contrib.layers.max_pool2d(
        inputs=conv,
        kernel_size=2,
        stride=2,
        padding='SAME')

    # Apply a Flatten Layer
    flat = tf.contrib.layers.flatten(inputs=conv)

    # Apply 1, 2, or 3 Fully Connected Layers
    batch, fc_num = flat.get_shape().as_list()
    fc = tf.contrib.layers.fully_connected(
        inputs=flat,
        num_outputs=fc_num,
        activation_fn=tf.nn.relu,
        weights_initializer=tf.random_normal_initializer,
        biases_initializer=tf.random_normal_initializer)

    fc = tf.nn.dropout(fc, keep_prob)

    # Apply an Output Layer (linear, no activation)
    out = tf.contrib.layers.fully_connected(
        inputs=fc,
        num_outputs=10,
        activation_fn=None,
        weights_initializer=tf.random_normal_initializer,
        biases_initializer=tf.random_normal_initializer)

    return out


def build_nn():
    print('Build Neural Network...')
    image_shape = (32, 32, 3)
    n_classes = 10

    # Remove previous weights, bias, inputs, etc..
    tf.reset_default_graph()

    # Input and output
    x = tf.placeholder(tf.float32, shape=(None,)+image_shape, name="x")
    y = tf.placeholder(tf.float32, shape=(None, n_classes), name="y")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    # Model
    logits = conv_net(x, keep_prob)

    # Name logits Tensor, so that is can be loaded from disk after training
    logits = tf.identity(logits, name='logits')

    # Loss and Optimizer
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(
        tf.cast(correct_pred, tf.float32), name='accuracy')

    return x, y, keep_prob, cost, optimizer, accuracy


def single_train(x, y, keep_prob, cost, optimizer, accuracy,
                 epochs, batch_size, keep_probability,
                 valid_features, valid_labels):
    print('\nChecking the Training on a Single Batch...')
    with tf.Session() as sess:
        # Initializing the variables
        sess.run(tf.global_variables_initializer())

        # Training cycle
        for epoch in range(epochs):
            start = time.time()
            batch_i = 1
            for batch_features, batch_labels in (
                    helper.load_preprocess_training_batch(batch_i, batch_size)):
                # Train neural network
                sess.run(optimizer, feed_dict={
                    x: batch_features,
                    y: batch_labels,
                    keep_prob: keep_probability})

            print('Epoch {:>2}, '.format(epoch + 1), end='')
            print('CIFAR-10 Batch {}:  '.format(batch_i), end='')
            loss = sess.run(cost, feed_dict={
                x: batch_features,
                y: batch_labels,
                keep_prob: 1.})
            valid_acc = sess.run(accuracy, feed_dict={
                x: valid_features,
                y: valid_labels,
                keep_prob: 1.})
            end = time.time()
            print('Loss: {:>10.4f} '.format(loss), end='')
            print('Validation Accuracy: {:.6f} '.format(valid_acc), end='')
            print('({:.1f} sec)'.format(end - start))


def full_train(x, y, keep_prob, cost, optimizer, accuracy,
               epochs, batch_size, keep_probability,
               valid_features, valid_labels):
    print('\nFull Training...')
    save_model_path = './image_classification'
    with tf.Session() as sess:
        # Initializing the variables
        sess.run(tf.global_variables_initializer())

        # Training cycle
        for epoch in range(epochs):
            # Loop over all batches
            n_batches = 5
            for batch_i in range(1, n_batches + 1):
                start = time.time()
                for batch_features, batch_labels in (
                        helper.load_preprocess_training_batch(
                            batch_i, batch_size)):
                    # Train neural network
                    sess.run(optimizer, feed_dict={
                        x: batch_features,
                        y: batch_labels,
                        keep_prob: keep_probability})

                print('Epoch {:>2}, '.format(epoch + 1), end='')
                print('CIFAR-10 Batch {}:  '.format(batch_i), end='')
                loss = sess.run(cost, feed_dict={
                    x: batch_features,
                    y: batch_labels,
                    keep_prob: 1.})
                valid_acc = sess.run(accuracy, feed_dict={
                    x: valid_features,
                    y: valid_labels,
                    keep_prob: 1.})
                end = time.time()
                print('Loss: {:>10.4f} '.format(loss), end='')
                print('Validation Accuracy: {:.6f} '.format(valid_acc), end='')
                print('({:.1f} sec)'.format(end - start))

        # Save Model
        saver = tf.train.Saver()
        save_path = saver.save(sess, save_model_path)


def test(batch_size):
    print('\nTesting...')
    save_model_path = './image_classification'
    n_samples = 4
    top_n_predictions = 3

    test_features, test_labels = pickle.load(
        open('preprocess_test.p', mode='rb'))
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('x:0')
        loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')

        # Get accuracy in batches for memory limitations
        test_batch_acc_total = 0
        test_batch_count = 0

        for test_feature_batch, test_label_batch in (
                helper.batch_features_labels(
                    test_features, test_labels, batch_size)):
            test_batch_acc_total += sess.run(
                loaded_acc,
                feed_dict={loaded_x: test_feature_batch,
                           loaded_y: test_label_batch,
                           loaded_keep_prob: 1.0})
            test_batch_count += 1

        print('Testing Accuracy: {}\n'.format(
            test_batch_acc_total/test_batch_count))

def main():
    DownloadDataSet()
    tests.test_folder_path(cifar10_dataset_folder_path)

    if not isfile('preprocess_validation.p'):
        helper.preprocess_and_save_data(
            cifar10_dataset_folder_path, normalize, one_hot_encode)
    valid_features, valid_labels = pickle.load(
        open('preprocess_validation.p', mode='rb'))

    x, y, keep_prob, cost, optimizer, accuracy = build_nn()

    epochs = 100
    batch_size = 1024
    keep_probability = 0.75

    need_training = True
    if need_training:
        if True:
            single_train(x, y, keep_prob, cost, optimizer, accuracy,
                         epochs, batch_size, keep_probability,
                         valid_features, valid_labels)

        full_train(x, y, keep_prob, cost, optimizer, accuracy,
                   epochs, batch_size, keep_probability,
                   valid_features, valid_labels)

    test(batch_size)


if __name__ == "__main__":
    main()
