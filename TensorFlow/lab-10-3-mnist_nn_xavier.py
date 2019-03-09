# Lab 10 MNIST and Xavier
import tensorflow as tf  # tensorflow
import random  # random
# import matplotlib.pyplot as plt  # matplotlib.pyplot
from tensorflow.examples.tutorials.mnist import input_data  # tensorflow mnist
tf.set_random_seed(777)  # 랜덤 시드 설정 reproducibility
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # mnist
# Check out https://www.tensorflow.org/get_started/mnist/beginners for more information about the mnist dataset
# 파라미터값들 parameters
learning_rate = 0.001  # 학습율
training_epochs = 15  # 에폭 횟수
batch_size = 100  # 배치 크기
# 입력 플레이스홀더 input place holders
X = tf.placeholder(tf.float32, [None, 784])  # X 플레이스홀더
Y = tf.placeholder(tf.float32, [None, 10])  # Y 플레이스홀더
# 신경망 레이어들을 위한 가중치와 바이어스 weights & bias for nn layers
# http://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
W1 = tf.get_variable("W1", shape=[784, 256], initializer=tf.contrib.layers.xavier_initializer())  # W1 변수 Xavier 초기화
b1 = tf.Variable(tf.random_normal([256]))  # b1 변수
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)  # Layer1
W2 = tf.get_variable("W2", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())  # W2 변수 Xavier 초기화
b2 = tf.Variable(tf.random_normal([256]))  # b2 변수
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)  # Layer2
W3 = tf.get_variable("W3", shape=[256, 10], initializer=tf.contrib.layers.xavier_initializer())  # W3 변수 Xavier 초기화
b3 = tf.Variable(tf.random_normal([10]))  # b3 변수
hypothesis = tf.matmul(L2, W3) + b3  # 가설 h
# 비용/손실/옵티마이저 정의 define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# 초기화 initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# 모델 훈련 train my model
for epoch in range(training_epochs):  # 에폭 반복
    avg_cost = 0  # 평균 비용
    total_batch = int(mnist.train.num_examples / batch_size)  # 배치 횟수
    for i in range(total_batch):  # 배치 반복
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # 배치 데이터
        feed_dict = {X: batch_xs, Y: batch_ys}  # 데이터 입력
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)  # cost, optimizer 계산
        avg_cost += c / total_batch  # 평균 비용 계산
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))  # epoch, avg_cost 출력
print('Learning Finished!')  # 학습 완료
# 모델 테스트 및 정확도 확인 Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))  # 가설과 Y가 일치하면 성공
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 일치한 평균이 정확도
print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))  # accuracy 출력
r = random.randint(0, mnist.test.num_examples - 1)  # 랜덤 데이터로 예측하기 Get one and predict
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))  # 랜덤 데이터 레이블
print("Prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))  # 랜덤 데이터 예측값

# plt.imshow(mnist.test.images[r:r + 1].
#           reshape(28, 28), cmap='Greys', interpolation='nearest')
# plt.show()

'''
Epoch: 0001 cost = 0.301498963
Epoch: 0002 cost = 0.107252513
Epoch: 0003 cost = 0.064888892
Epoch: 0004 cost = 0.044463030
Epoch: 0005 cost = 0.029951642
Epoch: 0006 cost = 0.020663404
Epoch: 0007 cost = 0.015853033
Epoch: 0008 cost = 0.011764387
Epoch: 0009 cost = 0.008598264
Epoch: 0010 cost = 0.007383116
Epoch: 0011 cost = 0.006839140
Epoch: 0012 cost = 0.004672963
Epoch: 0013 cost = 0.003979437
Epoch: 0014 cost = 0.002714260
Epoch: 0015 cost = 0.004707661
Learning Finished!
Accuracy: 0.9783
'''

'''
실행결과
Epoch: 0013 cost = 0.012618846
Epoch: 0014 cost = 0.008104711
Epoch: 0015 cost = 0.009540584
Learning Finished!
Accuracy: 0.9799
Label:  [7]
Prediction:  [7]
'''