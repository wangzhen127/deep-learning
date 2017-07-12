import helper
import numpy as np
from os.path import isfile, isdir
import pickle
import problem_unittests as tests
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
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
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
    return range_low + ( ( (x - min_value)*(range_high - range_low) )/( max_value - min_value ) )


def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    LABELS = np.array([0,1,2,3,4,5,6,7,8,9])
    lb = preprocessing.LabelBinarizer()
    lb.fit(LABELS)
    return lb.transform(x)


def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    batch, image_width, image_height, color_channels = x_tensor.get_shape().as_list()

    conv_filter_height = conv_ksize[0]
    conv_filter_width = conv_ksize[1]
    conv_strides_height = conv_strides[0]
    conv_strides_width = conv_strides[1]

    weight = tf.Variable(tf.truncated_normal(
        [conv_filter_height, conv_filter_width, color_channels, conv_num_outputs]))
    bias = tf.Variable(tf.zeros(conv_num_outputs))

    conv_layer = tf.nn.conv2d(
        x_tensor,
        weight,
        strides=[1, conv_strides_height, conv_strides_width, 1],
        padding='SAME')
    conv_layer = tf.nn.bias_add(conv_layer, bias)
    conv_layer = tf.nn.relu(conv_layer)

    pool_height = pool_ksize[0]
    pool_width = pool_ksize[1]
    pool_strides_height = pool_strides[0]
    pool_strides_width = pool_strides[1]

    conv_layer = tf.nn.max_pool(
        conv_layer,
        ksize=[1, pool_height, pool_width, 1],
        strides=[1, pool_strides_height, pool_strides_width, 1],
        padding='SAME')

    return conv_layer


def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    batch, image_width, image_height, color_channels = x_tensor.get_shape().as_list()
    return tf.reshape(x_tensor, [-1, image_width*image_height*color_channels])


def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    batch, num_inputs = x_tensor.get_shape().as_list()
    weight = tf.Variable(tf.random_normal([num_inputs, num_outputs]))
    bias = tf.Variable(tf.random_normal([num_outputs]))

    fc = tf.add(tf.matmul(x_tensor, weight), bias)
    fc = tf.nn.relu(fc)

    return fc


def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    batch, num_inputs = x_tensor.get_shape().as_list()
    weight = tf.Variable(tf.random_normal([num_inputs, num_outputs]))
    bias = tf.Variable(tf.random_normal([num_outputs]))

    return tf.add(tf.matmul(x_tensor, weight), bias)


def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    # Apply 1, 2, or 3 Convolution and Max Pool layers
    conv = conv2d_maxpool(x, 64, (3, 3), (1, 1), (2, 2), (2, 2))
    conv = conv2d_maxpool(conv, 128, (3, 3), (1, 1), (2, 2), (2, 2))

    # Apply a Flatten Layer
    flat = flatten(conv)

    # Apply 1, 2, or 3 Fully Connected Layers
    batch, fc_num = flat.get_shape().as_list()

    fc = fully_conn(flat, fc_num)
    fc = tf.nn.dropout(fc, keep_prob)

    # Apply an Output Layer
    out = output(fc, 10)

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


if __name__ == "__main__":
    main()
