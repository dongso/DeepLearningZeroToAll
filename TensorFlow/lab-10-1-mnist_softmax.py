# Lab 7 학습율과 평가 Learning rate and Evaluation
import tensorflow as tf  # tensorflow
import matplotlib.pyplot as plt  # matplotlib.pyplot
import random  # random
from tensorflow.examples.tutorials.mnist import input_data  # tensorflow mnist
tf.set_random_seed(777)  # 랜덤 시드 설정 reproducibility
# Check out https://www.tensorflow.org/get_started/mnist/beginners for more information about the mnist dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # mnist 데이터
# 입력 플레이스홀더 input place holders
X = tf.placeholder(tf.float32, [None, 784])  # X 플레이스홀더 
Y = tf.placeholder(tf.float32, [None, 10])  # Y 플레이스홀더
# 신경망 레이어들을 위한 가중치와 바이어스 weights & bias for nn layers
W = tf.Variable(tf.random_normal([784, 10]))  # W 변수
b = tf.Variable(tf.random_normal([10]))  # b 변수
# 파라미터 값들 parameters
learning_rate = 0.001  # 학습율
batch_size = 100  # 배치 크기
num_epochs = 50  # 에폭 횟수
num_iterations = int(mnist.train.num_examples / batch_size)  # 배치 반복 횟수
hypothesis = tf.matmul(X, W) + b  # 가설
# 비용/손실/옵티마이저 정의 define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=tf.stop_gradient(Y)))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # 비용 최소화
correct_prediction = tf.equal(tf.argmax(hypothesis, axis=1), tf.argmax(Y, axis=1))  # 가설 결과 최대값의 번호와 Y가 같으면 예측 성공
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 성공한 값의 평균값이 정확도
# 모델 훈련 train my model
with tf.Session() as sess:  # 세션
    sess.run(tf.global_variables_initializer())  # 변수 초기화 initialize
    for epoch in range(num_epochs):  # 에폭 반복
        avg_cost = 0  # 평균 비용 저장할 변수
        for iteration in range(num_iterations):  # 배치 반복수
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # 배치 데이터
            _, cost_val = sess.run([train, cost], feed_dict={X: batch_xs, Y: batch_ys})  # trsin, cost 계산
            avg_cost += cost_val / num_iterations  # 평균 비용 계산
        print(f"Epoch: {(epoch + 1):04d}, Cost: {avg_cost:.9f}")  # 에폭, 평균 비용 출력
    print("Learning Finished!")  # 학습 완료
    # 모델 테스트 및 정확도 확인 Test model and check accuracy
    print("Accuracy:",sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}),)  # accuracy 계산
    r = random.randint(0, mnist.test.num_examples - 1)  # 랜덤 데이터로 예측하기 Get one and predict
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r : r + 1], axis=1)))  # 랜덤 데이터의 레이블
    print("Prediction: ",sess.run(tf.argmax(hypothesis, axis=1), feed_dict={X: mnist.test.images[r : r + 1]}),)  # 랜덤 데이터의 예측값
    plt.imshow(mnist.test.images[r : r + 1].reshape(28, 28),cmap="Greys",interpolation="nearest",)  # 랜덤 데이터의 이미지
    plt.show()

'''
Epoch: 0001 Cost: 5.745170949
Epoch: 0002 Cost: 1.780056722
Epoch: 0003 Cost: 1.122778654
...
Epoch: 0048 Cost: 0.271918680
Epoch: 0049 Cost: 0.270640434
Epoch: 0050 Cost: 0.269054370
Learning Finished!
Accuracy: 0.9194
'''

'''
실행결과
Epoch: 0049, Cost: 0.270640430
Epoch: 0050, Cost: 0.269054366
Learning Finished!
Accuracy: 0.9194
Label:  [2]
Prediction:  [8]
'''
