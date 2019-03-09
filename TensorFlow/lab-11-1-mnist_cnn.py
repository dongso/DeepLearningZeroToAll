# Lab 11 MNIST및 컨볼루션 신경망 MNIST and Convolutional Neural Network
import tensorflow as tf  # tensorflow
import random  # random
# import matplotlib.pyplot as plt  # matplotlib.pyplot
from tensorflow.examples.tutorials.mnist import input_data  # tensorflow mnist
tf.set_random_seed(777)  # 랜덤 시드 설정 reproducibility
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # mnist
# Check out https://www.tensorflow.org/get_started/mnist/beginners for more information about the mnist dataset
# 하이퍼 파라미터들 hyper parameters
learning_rate = 0.001  # 학습율
training_epochs = 15  # 에폭 횟수
batch_size = 100  # 배치 크기
# 입력 플레이스홀더 input place holders
X = tf.placeholder(tf.float32, [None, 784])  # X 플레이스홀더
X_img = tf.reshape(X, [-1, 28, 28, 1])   # img 28x28x1 (black/white)  # X reshape
Y = tf.placeholder(tf.float32, [None, 10])  # Y 플레이스홀더
# L1 ImgIn shape=(?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))  # W1
#    Conv     -> (?, 28, 28, 32)
#    Pool     -> (?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')  # conv2d
L1 = tf.nn.relu(L1)  # relu
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # max pool
'''
Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
'''
# L2 ImgIn shape=(?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))  # W2
#    Conv      ->(?, 14, 14, 64)
#    Pool      ->(?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')  # conv2d
L2 = tf.nn.relu(L2)  # relu
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # max pool
L2_flat = tf.reshape(L2, [-1, 7 * 7 * 64])  # reshape
'''
Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
Tensor("Reshape_1:0", shape=(?, 3136), dtype=float32)
'''
# 최종 완전 연결망 Final FC 7x7x64 inputs -> 10 outputs
W3 = tf.get_variable("W3", shape=[7 * 7 * 64, 10], initializer=tf.contrib.layers.xavier_initializer())  # W2 xavier 초기화
b = tf.Variable(tf.random_normal([10]))  # b
logits = tf.matmul(L2_flat, W3) + b  # logits
# 비용/손실/옵티마이저 define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# 초기화 initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# 모델 훈련 train my model
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):  # 에폭 횟수
    avg_cost = 0  # 평균 비용
    total_batch = int(mnist.train.num_examples / batch_size)  # 배치 횟수
    for i in range(total_batch):  # 배치 반복
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # 배치 데이터
        feed_dict = {X: batch_xs, Y: batch_ys}  # feed_dict
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)  # cost, optimizer 계산
        avg_cost += c / total_batch  # 평균 비용 계산
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))  # epoch, avg_cost
print('Learning Finished!')  # 학습 완료
# 모델 검증 및 정확도 확인 Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
# 랜덤 데이터로 예측 Get one and predict
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(tf.argmax(logits, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

# plt.imshow(mnist.test.images[r:r + 1].
#           reshape(28, 28), cmap='Greys', interpolation='nearest')
# plt.show()

'''
Epoch: 0001 cost = 0.340291267
Epoch: 0002 cost = 0.090731326
Epoch: 0003 cost = 0.064477619
Epoch: 0004 cost = 0.050683064
Epoch: 0005 cost = 0.041864835
Epoch: 0006 cost = 0.035760704
Epoch: 0007 cost = 0.030572132
Epoch: 0008 cost = 0.026207981
Epoch: 0009 cost = 0.022622454
Epoch: 0010 cost = 0.019055919
Epoch: 0011 cost = 0.017758641
Epoch: 0012 cost = 0.014156652
Epoch: 0013 cost = 0.012397016
Epoch: 0014 cost = 0.010693789
Epoch: 0015 cost = 0.009469977
Learning Finished!
Accuracy: 0.9885
'''

'''
실행결과
Accuracy: 0.9883
Label:  [6]
Prediction:  [6]
'''