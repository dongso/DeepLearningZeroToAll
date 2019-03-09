# Lab 7 학습율과 평가 Learning rate and Evaluation
import tensorflow as tf  # tensorflow
import matplotlib.pyplot as plt  # matplotlib.pyplot
import random  # random
tf.set_random_seed(777)  # 랜덤 시드 설정 for reproducibility
from tensorflow.examples.tutorials.mnist import input_data  # tensorflow mnist 데이터셋
# Check out https://www.tensorflow.org/get_started/mnist/beginners for more information about the mnist dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # mnist 데이터셋
nb_classes = 10  # 10종류 분류
X = tf.placeholder(tf.float32, [None, 784])  # MNIST 데이터는 28 * 28 크기의 이미지 MNIST data image of shape 28 * 28 = 784
Y = tf.placeholder(tf.float32, [None, nb_classes])  # 0~9의 숫자를 인식하는 10 클래스. 0 - 9 digits recognition = 10 classes
W = tf.Variable(tf.random_normal([784, nb_classes]))  # W 변수
b = tf.Variable(tf.random_normal([nb_classes]))  # b 변수
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)  # 소프트맥스를 사용하는 가설 Hypothesis (using softmax)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))  # 비용함수
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)  # 경사하강법 비용 최소화
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))  # 모델 평가 Test model
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))  # 정확도 계산 Calculate accuracy
# 파라미터값들 parameters
num_epochs = 15  # 에폭 : 데이터셋을 몇번 반복학습하는 횟수
batch_size = 100  # 배치 : 데이터셋을 나누어 반복하는 단위 크기
num_iterations = int(mnist.train.num_examples / batch_size)  # 반복수 : 데이터 수에 대해 배치 크기로 반복할 횟수
with tf.Session() as sess:  # 세션
    sess.run(tf.global_variables_initializer())  # 변수 초기화 Initialize TensorFlow variables
    for epoch in range(num_epochs):  # 반복적 학습 Training cycle
        avg_cost = 0  # 평균 비용 저장할 변수
        for i in range(num_iterations):  # 배치 반복
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # batch size 만큼 batch_xs, batch_ys 가져옴
            _, cost_val = sess.run([train, cost], feed_dict={X: batch_xs, Y: batch_ys})  # feed_dict 로 입력하고 train, cost 실행
            avg_cost += cost_val / num_iterations  # 배치에 대한 cost 를 avg cost 로 계산
        print("Epoch: {:04d}, Cost: {:.9f}".format(epoch + 1, avg_cost))  # epoch, cost 출력
    print("Learning finished")  # 학습 완료
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}),)  # 테스트셋으로 모델 평가 Test the model using test sets
    r = random.randint(0, mnist.test.num_examples - 1)  # 랜덤 데이터로 예측하기 Get one and predict
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r : r + 1], 1)))  # 랜덤 데이터의 실제 정답
    print("Prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r : r + 1]}),)  # 랜덤 데이터의 예측 값
    plt.imshow(mnist.test.images[r : r + 1].reshape(28, 28), cmap="Greys", interpolation="nearest",)  # 랜덤 데이터에 대한 이미지 출력
    plt.show()

'''
Epoch: 0001, Cost: 2.826302672
Epoch: 0002, Cost: 1.061668952
Epoch: 0003, Cost: 0.838061315
Epoch: 0004, Cost: 0.733232745
Epoch: 0005, Cost: 0.669279885
Epoch: 0006, Cost: 0.624611836
Epoch: 0007, Cost: 0.591160344
Epoch: 0008, Cost: 0.563868987
Epoch: 0009, Cost: 0.541745171
Epoch: 0010, Cost: 0.522673578
Epoch: 0011, Cost: 0.506782325
Epoch: 0012, Cost: 0.492447643
Epoch: 0013, Cost: 0.479955837
Epoch: 0014, Cost: 0.468893674
Epoch: 0015, Cost: 0.458703488
Learning finished
Accuracy:  0.8951
'''

'''
실행결과
Epoch: 0013, Cost: 0.479955841
Epoch: 0014, Cost: 0.468893681
Epoch: 0015, Cost: 0.458703478
Learning finished
Accuracy:  0.8951
Label:  [1]
Prediction:  [1]
'''