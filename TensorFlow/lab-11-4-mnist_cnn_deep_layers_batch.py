import tensorflow as tf, os, random
from tensorflow.examples.tutorials.mnist import input_data
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.set_random_seed(777)
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

learning_rate = 0.001
training_epochs = 15
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

        conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], padding='SAME', strides=2)
        dropout1 = tf.layers.dropout(inputs=pool1, rate=0.3, training=self.training)

        conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], padding="SAME", strides=2)
        dropout2 = tf.layers.dropout(inputs=pool2, rate=0.3, training=self.training)

        conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], padding="same", strides=2)
        dropout3 = tf.layers.dropout(inputs=pool3, rate=0.3, training=self.training)

        flat = tf.reshape(dropout3, [-1, 4 * 4 * 128])
        dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu)
        dropout4 = tf.layers.dropout(inputs=dense4, rate=0.5, training=self.training)

        self.logits = tf.layers.dense(inputs=dropout4, units=10)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.total_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(tf.argmax(self.logits, 1), feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.total_correct, feed_dict={self.X: x_test, self.Y: y_test, self.training: training})

    def train(self, x_train, y_train, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_train, self.Y: y_train, self.training: training})

sess = tf.Session()
m1 = Model(sess, 'm1')
sess.run(tf.global_variables_initializer())
total_batch = int(mnist.train.num_examples / batch_size)
for epoch in range(training_epochs):
    avg_cost = 0
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c, _ = m1.train(batch_xs, batch_ys)
        avg_cost += c / total_batch
    print('Epoch:', '%04d' % (epoch + 1), 'cost:', '{:.9f}'.format(avg_cost))
correct_sum = 0
for i in range(total_batch):
    batch_xs, batch_ys = mnist.test.next_batch(batch_size)
    batch_correct_count = m1.get_accuracy(batch_xs, batch_ys)
    correct_sum += batch_correct_count
total_accuracy = correct_sum / total_batch
print('Accuracy:', total_accuracy)

r = random.randint(0, mnist.test.num_examples - 1)
print('Label: ', sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print('Prediction: ', m1.predict(mnist.test.images[r:r + 1]))