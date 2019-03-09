# Lab 11 MNIST와 딥러닝 CNN. MNIST and Deep learning CNN
# https://www.tensorflow.org/tutorials/layers
import tensorflow as tf  # tensorflow
import numpy as np  # numpy
from tensorflow.examples.tutorials.mnist import input_data  # tensorflow mnist
tf.set_random_seed(777)  # 랜덤 시드 설정 reproducibility
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # mnist
# Check out https://www.tensorflow.org/get_started/mnist/beginners for more information about the mnist dataset
# 하이퍼파라미터값들 hyper parameters
learning_rate = 0.001  # 학습율
training_epochs = 20  # 에폭 횟수
batch_size = 100  # 배치 크기

class Model: # 모델 클래스 

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

            conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)  # Convolutional Layer #3 and Pooling Layer #3
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], padding="SAME", strides=2)
            dropout3 = tf.layers.dropout(inputs=pool3, rate=0.3, training=self.training)

            flat = tf.reshape(dropout3, [-1, 128 * 4 * 4])
            dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu)  # Dense Layer with Relu
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
models = []  # 모델 리스트
num_models = 2  # 모델수
for m in range(num_models):  # 모델 생성
    models.append(Model(sess, "model" + str(m)))
sess.run(tf.global_variables_initializer())
print('Learning Started!')
# 학습 train my model
for epoch in range(training_epochs):  # 에폭 반복
    avg_cost_list = np.zeros(len(models))  # 평균 비용 리스트
    total_batch = int(mnist.train.num_examples / batch_size)  # 배치 횟수
    for i in range(total_batch):  # 배치 반복
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # 배치 데이터
        # 각 모델 학습 train each model
        for m_idx, m in enumerate(models):  # 각 모델에 대해 반복
            c, _ = m.train(batch_xs, batch_ys)  # 학습
            avg_cost_list[m_idx] += c / total_batch  # 평균 비용 리스트
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', avg_cost_list)
print('Learning Finished!')
# 모델 검증 및 정확도 확인 Test model and check accuracy
test_size = len(mnist.test.labels)
predictions = np.zeros([test_size, 10])
for m_idx, m in enumerate(models):  # 모델 반복
    print(m_idx, 'Accuracy:', m.get_accuracy(mnist.test.images, mnist.test.labels))  # test 데이터 정확도 계산 
    p = m.predict(mnist.test.images)  # 예측
    predictions += p  # 예측값 합산
ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(mnist.test.labels, 1))  # 앙상블 예측과 정답의 일치
ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))  # 앙상블 정확도
print('Ensemble accuracy:', sess.run(ensemble_accuracy))

'''
0 Accuracy: 0.9933
1 Accuracy: 0.9946
2 Accuracy: 0.9934
3 Accuracy: 0.9935
4 Accuracy: 0.9935
5 Accuracy: 0.9949
6 Accuracy: 0.9941

Ensemble accuracy: 0.9952
'''
