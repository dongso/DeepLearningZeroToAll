# Lab 11 MNIST와 딥러닝 CNN. MNIST and Deep learning CNN
import tensorflow as tf  # tensorflow
# import matplotlib.pyplot as plt  # matplotlib.pyplot
from tensorflow.examples.tutorials.mnist import input_data  # tensorflow mnist
tf.set_random_seed(777)  # 랜덤 시드 설정 reproducibility
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # mnist
# Check out https://www.tensorflow.org/get_started/mnist/beginners for more information about the mnist dataset
# 하이퍼파라미터값들 hyper parameters
learning_rate = 0.001  # 학습율
training_epochs = 15  # 에폭 횟수
batch_size = 100  # 배치 크기
class Model:  # 모델 클래스
    def __init__(self, sess, name):
        self.sess = sess  # 세션
        self.name = name  # 이름
        self._build_net()  # 신경망

    def _build_net(self):
        with tf.variable_scope(self.name):
            # 드롭아웃 비율. dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
            self.training = tf.placeholder(tf.bool)
            # 입력 플레이스홀더 input place holders
            self.X = tf.placeholder(tf.float32, [None, 784])
            # 이미지 img 28x28x1 (black/white), Input Layer
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)  # Convolutional Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], padding="SAME", strides=2)  # Pooling Layer #1
            dropout1 = tf.layers.dropout(inputs=pool1, rate=0.3, training=self.training)

            conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)  # Convolutional Layer #2 and Pooling Layer #2
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], padding="SAME", strides=2)
            dropout2 = tf.layers.dropout(inputs=pool2, rate=0.3, training=self.training)

            conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)  # Convolutional Layer #2 and Pooling Layer #2
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], padding="same", strides=2)
            dropout3 = tf.layers.dropout(inputs=pool3, rate=0.3, training=self.training)

            flat = tf.reshape(dropout3, [-1, 128 * 4 * 4])
            dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu) # Dense Layer with Relu
            dropout4 = tf.layers.dropout(inputs=dense4, rate=0.5, training=self.training)

            self.logits = tf.layers.dense(inputs=dropout4, units=10)  # Logits (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs

        # 비용/손실/옵티마이저 define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):  # 예측
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):  # 정확도
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):  # 학습
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.training: training})

# 초기화 initialize
sess = tf.Session()
m1 = Model(sess, "m1")
sess.run(tf.global_variables_initializer())
print('Learning Started!')

# 학습 train my model
for epoch in range(training_epochs):  # 에폭 반복
    avg_cost = 0  # 평균 비용
    total_batch = int(mnist.train.num_examples / batch_size)  # 배치 횟수
    for i in range(total_batch):  # 배치 반복
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # 배치 데이터
        c, _ = m1.train(batch_xs, batch_ys)  # 학습
        avg_cost += c / total_batch  # 평균 비용
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
print('Learning Finished!')
# 모델 검증 및 정확도 확인 Test model and check accuracy
print('Accuracy:', m1.get_accuracy(mnist.test.images, mnist.test.labels))
