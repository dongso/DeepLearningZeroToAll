import time
start_time = time.time()

import tensorflow as tf, os, random, numpy as np
from tensorflow.examples.tutorials.mnist import input_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.set_random_seed(777)
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

learning_rate = 0.001
training_epochs = 20
batch_size = 100

class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        self.training = tf.placeholder(tf.bool)
        self.X = tf.placeholder(tf.float32, [None, 784])
        X_img = tf.reshape(self.X, [-1, 28, 28, 1])
        self.Y = tf.placeholder(tf.float32, [None, 10])

        conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3], kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='SAME', activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], padding='SAME', strides=2)
        dropout1 = tf.layers.dropout(inputs=pool1, rate=0.3, training=self.training)

        conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3], kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="SAME", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], padding="SAME", strides=2)
        dropout2 = tf.layers.dropout(inputs=pool2, rate=0.3, training=self.training)

        conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3], kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same", activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], padding="same", strides=2)
        dropout3 = tf.layers.dropout(inputs=pool3, rate=0.3, training=self.training)

        flat = tf.reshape(dropout3, [-1, 4 * 4 * 128])
        dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        dropout4 = tf.layers.dropout(inputs=dense4, rate=0.5, training=self.training)

        self.logits = tf.layers.dense(inputs=dropout4, units=10, kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.total_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.total_correct, feed_dict={self.X: x_test, self.Y: y_test, self.training: training})

    def train(self, x_train, y_train, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_train, self.Y: y_train, self.training: training})

    def evaluate(self, X_sample, y_sample, batch_size=batch_size):
        N = X_sample.shape[0]
        correct_sample = 0
        for i in range(0, N, batch_size):
            X_batch = X_sample[i: i + batch_size]
            y_batch = y_sample[i: i + batch_size]
            N_batch = X_batch.shape[0]
            correct_sample += self.get_accuracy(X_batch, y_batch, training=False) * N_batch
        return correct_sample / N

sess = tf.Session()
models = []
num_models = 7
for m in range(num_models):
    models.append(Model(sess, "model" + str(m)))
sess.run(tf.global_variables_initializer())
train_total_batch = int(mnist.train.num_examples / batch_size)
for epoch in range(training_epochs):
    avg_cost_list = np.zeros(len(models))
    for i in range(train_total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        for m_idx, m in enumerate(models):
            c, _ = m.train(batch_xs, batch_ys)
            avg_cost_list[m_idx] += c / train_total_batch
    print('Epoch:', '%04d' % (epoch + 1), 'cost:', avg_cost_list)

test_total_batch = int(mnist.test.num_examples / batch_size)
test_size = len(mnist.test.labels)
predictions = np.zeros([test_size, 10])
for i in range(test_total_batch):
    prediction_batch = np.zeros([batch_size, 10])
    accuracy_batch = np.zeros([len(models)])
    batch_xs, batch_ys = mnist.test.next_batch(batch_size)
    for m_idx, m in enumerate(models):
        accuracy_batch[m_idx] += m.get_accuracy(batch_xs, batch_ys)
        p = m.predict(batch_xs)
        prediction_batch += p
    predictions[i * batch_size:i * batch_size + batch_size] = prediction_batch
    print(i, ' : Accuracy : ', accuracy_batch)
ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(mnist.test.labels, 1))
ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
print('Ensemble accuracy:', sess.run(ensemble_accuracy))

end_time = time.time()
print('elapsed time : ', end_time - start_time)
