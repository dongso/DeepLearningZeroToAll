# Lab 11 MNIST와 딥러닝 CNN. MNIST and Deep learning CNN
import tensorflow as tf  # tensorflow
import random  # random
# import matplotlib.pyplot as plt  # matplotlib.pyplot
from tensorflow.examples.tutorials.mnist import input_data  # tensorflow mnist
tf.set_random_seed(777)  # 랜덤 시드 설정 reproducibility
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # mnist
# Check out https://www.tensorflow.org/get_started/mnist/beginners for more information about the mnist dataset
# 하이퍼 파라미터값들 hyper parameters
learning_rate = 0.001  # 학습율
training_epochs = 15  # 에폭 횟수
batch_size = 100  # 배치 크기
# 드롭아웃 비율. dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)
# 입력 플레이스홀더 input place holders
X = tf.placeholder(tf.float32, [None, 784])  # X 플레이스홀더
X_img = tf.reshape(X, [-1, 28, 28, 1])   # img 28x28x1 (black/white)  # reshape
Y = tf.placeholder(tf.float32, [None, 10])  # Y 플레이스홀더
# L1 ImgIn shape=(?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))  # W1
#    Conv     -> (?, 28, 28, 32)
#    Pool     -> (?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')  # conv2d
L1 = tf.nn.relu(L1)  # relu
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # max pool
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)  # dropout
'''
Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
Tensor("dropout/mul:0", shape=(?, 14, 14, 32), dtype=float32)
'''
# L2 ImgIn shape=(?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))  # W2
#    Conv      ->(?, 14, 14, 64)
#    Pool      ->(?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')  # conv2d
L2 = tf.nn.relu(L2)  # relu
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # max pool
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)  # dropout
'''
Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
Tensor("dropout_1/mul:0", shape=(?, 7, 7, 64), dtype=float32)
'''
# L3 ImgIn shape=(?, 7, 7, 64)
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))  # W3
#    Conv      ->(?, 7, 7, 128)
#    Pool      ->(?, 4, 4, 128)
#    Reshape   ->(?, 4 * 4 * 128) # Flatten them for FC
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')  # conv2d
L3 = tf.nn.relu(L3)  # relu
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # max pool
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)  # dropout
L3_flat = tf.reshape(L3, [-1, 128 * 4 * 4])  # flat reshape
'''
Tensor("Conv2D_2:0", shape=(?, 7, 7, 128), dtype=float32)
Tensor("Relu_2:0", shape=(?, 7, 7, 128), dtype=float32)
Tensor("MaxPool_2:0", shape=(?, 4, 4, 128), dtype=float32)
Tensor("dropout_2/mul:0", shape=(?, 4, 4, 128), dtype=float32)
Tensor("Reshape_1:0", shape=(?, 2048), dtype=float32)
'''
# L4 FC 4x4x128 inputs -> 625 outputs
W4 = tf.get_variable("W4", shape=[128 * 4 * 4, 625], initializer=tf.contrib.layers.xavier_initializer())  # W4 xavier 초기화
b4 = tf.Variable(tf.random_normal([625]))  # b4
L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)  # relu
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)  # dropout
'''
Tensor("Relu_3:0", shape=(?, 625), dtype=float32)
Tensor("dropout_3/mul:0", shape=(?, 625), dtype=float32)
'''
# L5 Final FC 625 inputs -> 10 outputs
W5 = tf.get_variable("W5", shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer())  # W5 xavier 초기화
b5 = tf.Variable(tf.random_normal([10]))  # b5
logits = tf.matmul(L4, W5) + b5  # logits
'''
Tensor("add_1:0", shape=(?, 10), dtype=float32)
'''
# 비용/손실/옵티마이저 define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# 초기화 initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# 모델 훈련 train my model
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):  # 에폭 반복
    avg_cost = 0  # 평균 비용
    total_batch = int(mnist.train.num_examples / batch_size)  # 배치 횟수
    for i in range(total_batch):  # 배치 반복
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # 배치 데이터
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}  # feed_dict
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)  # cost, optimizer 계산
        avg_cost += c / total_batch  # 평균 비용 계산
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))  # epoch, avg_cost
print('Learning Finished!')  # 학습 완료
# 모델 검증 및 정확성 확인 Test model and check accuracy
# if you have a OOM error, please refer to lab-11-X-mnist_deep_cnn_low_memory.py
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))
# 랜덤 데이터로 예측 Get one and predict
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(tf.argmax(logits, 1), feed_dict={X: mnist.test.images[r:r + 1], keep_prob: 1}))

# plt.imshow(mnist.test.images[r:r + 1].
#           reshape(28, 28), cmap='Greys', interpolation='nearest')
# plt.show()

'''
Learning stared. It takes sometime.
Epoch: 0001 cost = 0.385748474
Epoch: 0002 cost = 0.092017397
Epoch: 0003 cost = 0.065854684
Epoch: 0004 cost = 0.055604566
Epoch: 0005 cost = 0.045996377
Epoch: 0006 cost = 0.040913645
Epoch: 0007 cost = 0.036924479
Epoch: 0008 cost = 0.032808939
Epoch: 0009 cost = 0.031791007
Epoch: 0010 cost = 0.030224456
Epoch: 0011 cost = 0.026849916
Epoch: 0012 cost = 0.026826763
Epoch: 0013 cost = 0.027188021
Epoch: 0014 cost = 0.023604777
Epoch: 0015 cost = 0.024607201
Learning Finished!
Accuracy: 0.9938
'''

'''
실행결과 
Epoch: 0001 cost = 0.371345862
Epoch: 0002 cost = 0.099684447
Epoch: 0003 cost = 0.073611460
Epoch: 0004 cost = 0.060713852
Epoch: 0005 cost = 0.054853792
Epoch: 0006 cost = 0.046815755
Epoch: 0007 cost = 0.043182768
'''