# Lab 5 로지스틱 회귀 분류기 Logistic Regression Classifier
import tensorflow as tf  # tensorflow
import numpy as np  # numpy
tf.set_random_seed(777)  # 랜덤 시드 설정 for reproducibility
xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)  # numpy로 csv 데이터 파일 읽기
x_data = xy[:, 0:-1]  # 마지막 열 전까지 X
y_data = xy[:, [-1]]  # 마지막 열만 Y
print(x_data.shape, y_data.shape)  # x, y 데이터 shape 출력
# feed_dict 하게될 텐서 플레이스홀더 placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 8])  # X 플레이스홀더
Y = tf.placeholder(tf.float32, shape=[None, 1])  # Y 플레이스홀더
W = tf.Variable(tf.random_normal([8, 1]), name='weight')  # W 변수
b = tf.Variable(tf.random_normal([1]), name='bias')  # b 변수
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)  # 시그모이드 함수를 사용한 가설 정의 Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(-tf.matmul(X, W)))
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))  # 비용/손실 함수 cost/loss function
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)  # 경사하강법 비용 최소화
# 정확도 계산 Accuracy computation
# 가설이 0.5 보다 크면 참이고 아니면 거짓 True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
with tf.Session() as sess:  # 그래프 실행 Launch graph
    sess.run(tf.global_variables_initializer())  # 변수 초기화 Initialize TensorFlow variables
    for step in range(10001):  # 10000회 반복
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})  # cost, train 실행
        if step % 200 == 0:  # 200회 마다 
            print(step, cost_val)  # step, cost 출력
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})  # 정확도 출력 Accuracy report
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)  # 가설 h, 일치 Y, 정확도 a

'''
0 0.82794
200 0.755181
400 0.726355
600 0.705179
800 0.686631
...
9600 0.492056
9800 0.491396
10000 0.490767

...

 [ 1.]
 [ 1.]
 [ 1.]]
Accuracy:  0.762846
'''

'''
실행결과
[1.]
 [1.]
 [1.]
 [1.]] 
Accuracy:  0.7628459
'''