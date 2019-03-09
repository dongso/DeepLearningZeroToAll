# Lab 9 XOR
import tensorflow as tf  # tensorflow
import numpy as np  # numpy
tf.set_random_seed(777)  # 랜덤 시드 설정 for reproducibility
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)  # x 데이터
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)  # y 데이터
X = tf.placeholder(tf.float32, [None, 2])  # X 플레이스홀더
Y = tf.placeholder(tf.float32, [None, 1])  # Y 플레이스홀더
W1 = tf.Variable(tf.random_normal([2, 2]), name='weight1')  # W1 변수
b1 = tf.Variable(tf.random_normal([2]), name='bias1')  # b1 변수
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)  # layer1은 기존 가설을 시그모이드 활성화 함수 사용한 결과
W2 = tf.Variable(tf.random_normal([2, 1]), name='weight2')  # W2 변수
b2 = tf.Variable(tf.random_normal([1]), name='bias2')  # b2 변수
hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)  # 가설 h는 layer1을 w2, b2와 활성화 함수 사용한 결과
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))  # 비용/손실 함수 cost/loss function
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)  # 경사하강법 비용 최소화
# 정확도 계산 Accuracy computation
# 0.5보다 크면 참이고, 아니면 거짓 True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
with tf.Session() as sess:  # 그래프 실행 Launch graph
    sess.run(tf.global_variables_initializer())  # 변수 초기화 Initialize TensorFlow variables
    for step in range(10001):  # 10000회 반복
        _, cost_val = sess.run([train, cost], feed_dict={X: x_data, Y: y_data})  # train, cost 실행
        if step % 100 == 0:  # 100회 마다
            print(step, cost_val)  # step, cost 실행
    h, p, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})  # 정확도 출력 Accuracy report
    print(f"\nHypothesis:\n{h} \nPredicted:\n{p} \nAccuracy:\n{a}")  # h, p, a 출력

'''
Hypothesis:
[[0.01338216]
 [0.98166394]
 [0.98809403]
 [0.01135799]] 
Predicted:
[[0.]
 [1.]
 [1.]
 [0.]] 
Accuracy:
1.0
'''

'''
실행결과
9800 0.014254246
9900 0.014047621
10000 0.013846756

Hypothesis:
[[0.01338216]
 [0.98166394]
 [0.98809403]
 [0.01135799]] 
Predicted:
[[0.]
 [1.]
 [1.]
 [0.]] 
Accuracy:
1.0
'''