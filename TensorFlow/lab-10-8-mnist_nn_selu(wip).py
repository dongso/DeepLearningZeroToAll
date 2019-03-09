# Lab 10 MNIST와 드롭아웃 MNIST and Dropout
# SELU implementation from https://github.com/bioinf-jku/SNNs/blob/master/selu.py
import tensorflow as tf  # tensorflow
import random  # random
# import matplotlib.pyplot as plt  # matplotlib.pyplot
# -*- coding: utf-8 -*-
'''
Tensorflow Implementation of the Scaled ELU function and Dropout
스케일된 ELU (SELU) 함수와 드롭아웃의 텐서플로 구현
'''
import numbers  # numbers
from tensorflow.contrib import layers  # layers
from tensorflow.python.framework import ops  # ops : Operation
from tensorflow.python.framework import tensor_shape  # tensor_shape
from tensorflow.python.framework import tensor_util  # tensor_util
from tensorflow.python.ops import math_ops  # math_ops
from tensorflow.python.ops import random_ops  # random_ops
from tensorflow.python.ops import array_ops  # array_ops
from tensorflow.python.layers import utils  # utils
from tensorflow.examples.tutorials.mnist import input_data  # tensorflow mnist
tf.set_random_seed(777)  # 랜덤 시드 설정 reproducibility
def selu(x):  # selu 함수
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

def dropout_selu(x, keep_prob, alpha= -1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0,
                 noise_shape=None, seed=None, name=None, training=False):  # 드롭아웃 selu 함수
    """Dropout to a value with rescaling."""

    def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
        keep_prob = 1.0 - rate  # 드롭아웃 비율
        x = ops.convert_to_tensor(x, name="x")  # 텐서로 변환
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:  # 적절한 범위가 아니면 에러 발생
            raise ValueError("keep_prob must be a scalar tensor or a float in the range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")  # 텐서로 변환
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())
        alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")  # 텐서로 변환
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())
        if tensor_util.constant_value(keep_prob) == 1:  # keep_prob 1이면 리턴
            return x
        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)  # noise_shape None 이면 array_ops.shape
        random_tensor = keep_prob  # 랜덤 텐서
        random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)  # uniform 분포의 랜덤값 추가
        binary_tensor = math_ops.floor(random_tensor)  # 크지 않은 최대 정수값
        ret = x * binary_tensor + alpha * (1-binary_tensor)
        a = tf.sqrt(fixedPointVar / (keep_prob *((1-keep_prob) * tf.pow(alpha-fixedPointMean,2) + fixedPointVar)))
        b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        ret = a * ret + b
        ret.set_shape(x.get_shape())
        return ret

    with ops.name_scope(name, "dropout", [x]) as name:
        return utils.smart_cond(training,
                                lambda: dropout_selu_impl(x, keep_prob, alpha, noise_shape, seed, name),
                                lambda: array_ops.identity(x))

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # mnist
# Check out https://www.tensorflow.org/get_started/mnist/beginners for more information about the mnist dataset
# 파라미터값들 parameters
learning_rate = 0.001  # 학습율
training_epochs = 50  # 에폭 횟수
batch_size = 100  # 배치 크기
# 입력 플레이스홀더 input place holders
X = tf.placeholder(tf.float32, [None, 784])  # X 플레이스홀더
Y = tf.placeholder(tf.float32, [None, 10])  # Y 플레이스홀더
# 드롭아웃 학습시 0.7, 검증시 1. dropout (keep_prob) rate  0.7 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)
# 신경망 레이어들을 위한 가중치와 바이어스. weights & bias for nn layers
# http://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
W1 = tf.get_variable("W1", shape=[784, 512], initializer=tf.contrib.layers.xavier_initializer())  # W1
b1 = tf.Variable(tf.random_normal([512]))  # b1
L1 = selu(tf.matmul(X, W1) + b1)  # L1 selu
L1 = dropout_selu(L1, keep_prob=keep_prob)  # L1 dropout
W2 = tf.get_variable("W2", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())  # W2
b2 = tf.Variable(tf.random_normal([512]))  # b2
L2 = selu(tf.matmul(L1, W2) + b2)  # L2 selu
L2 = dropout_selu(L2, keep_prob=keep_prob)  # L2 dropout
W3 = tf.get_variable("W3", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())  # W3
b3 = tf.Variable(tf.random_normal([512]))  # b3
L3 = selu(tf.matmul(L2, W3) + b3)  # L3 selu
L3 = dropout_selu(L3, keep_prob=keep_prob)  # L3 dropout
W4 = tf.get_variable("W4", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())  # W4
b4 = tf.Variable(tf.random_normal([512]))  # b4
L4 = selu(tf.matmul(L3, W4) + b4)  # L4 selu
L4 = dropout_selu(L4, keep_prob=keep_prob)  # L4 dropout
W5 = tf.get_variable("W5", shape=[512, 10], initializer=tf.contrib.layers.xavier_initializer())  # W5
b5 = tf.Variable(tf.random_normal([10]))  # b5
hypothesis = tf.matmul(L4, W5) + b5  # 가설 h
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))  # 비용/손실/옵티마이저 define cost/loss & optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # 최소화
# 초기화 initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# 모델 훈련 train my model
for epoch in range(training_epochs):  # 에폭 반복
    avg_cost = 0  # 평균 비용
    total_batch = int(mnist.train.num_examples / batch_size)  # 배치 횟수
    for i in range(total_batch):  # 배치 반복
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # 배치 데이터
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}  # feed_dict 
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)  # cost, optimizer 계산
        avg_cost += c / total_batch  # 평균 비용
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))  # epoch, avg_cost
print('Learning Finished!')  # 학습 완료
# 모델 검증 및 정확도 확인 Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))
# 랜덤 데이터로 예측 Get one and predict
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1], keep_prob: 1}))

# plt.imshow(mnist.test.images[r:r + 1].
#           reshape(28, 28), cmap='Greys', interpolation='nearest')
# plt.show()

'''
Epoch: 0001 cost = 0.447322626
Epoch: 0002 cost = 0.157285590
Epoch: 0003 cost = 0.121884535
Epoch: 0004 cost = 0.098128681
Epoch: 0005 cost = 0.082901778
Epoch: 0006 cost = 0.075337573
Epoch: 0007 cost = 0.069752543
Epoch: 0008 cost = 0.060884363
Epoch: 0009 cost = 0.055276413
Epoch: 0010 cost = 0.054631256
Epoch: 0011 cost = 0.049675195
Epoch: 0012 cost = 0.049125314
Epoch: 0013 cost = 0.047231930
Epoch: 0014 cost = 0.041290121
Epoch: 0015 cost = 0.043621063
Learning Finished!
Accuracy: 0.9804
'''

'''
실행결과
Epoch: 0048 cost = 0.025589629
Epoch: 0049 cost = 0.014068754
Epoch: 0050 cost = 0.028472713
Learning Finished!
Accuracy: 0.9811
Label:  [6]
Prediction:  [6]
'''