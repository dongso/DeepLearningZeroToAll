# Lab 4 다중 변수 선형 회귀 Multi-variable linear regression
import tensorflow as tf  # tensorflow
import numpy as np  # numpy
tf.set_random_seed(777)  # 랜덤 시드 설정 for reproducibility
xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)  # 콤마 구분 csv 데이터 파일 불러오기
x_data = xy[:, 0:-1]  # 마지막 전까지 x 데이터
y_data = xy[:, [-1]]  # 마지막 부분만 y 데이터
# 불러온 데이터의 모양을 확인한다. Make sure the shape and data are OK
print(x_data, "\nx_data shape:", x_data.shape)  # x 데이터의 shape 출력
print(y_data, "\ny_data shape:", y_data.shape)  # x 데이터의 shape 출력
# 데이터 출력 data output
'''
[[ 73.  80.  75.]
 [ 93.  88.  93.]
 ...
 [ 76.  83.  71.]
 [ 96.  93.  95.]] 
x_data shape: (25, 3)
[[152.]
 [185.]
 ...
 [149.]
 [192.]] 
y_data shape: (25, 1)
'''
# feed_dict 하게될 텐서 플레이스홀더 placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])  # X 플레이스홀더
Y = tf.placeholder(tf.float32, shape=[None, 1])  # Y 플레이스홀더
W = tf.Variable(tf.random_normal([3, 1]), name='weight')  # W 변수
b = tf.Variable(tf.random_normal([1]), name='bias')  # b 변수
hypothesis = tf.matmul(X, W) + b  # 가설 Hypothesis
cost = tf.reduce_mean(tf.square(hypothesis - Y))  # 비용/손실 함수 Simplified cost/loss function
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)  # 최소화 Minimize
train = optimizer.minimize(cost)
sess = tf.Session()  # 세션에서 그래프 실행 Launch the graph in a session.
sess.run(tf.global_variables_initializer())  # 그래프에서 변수 초기화 Initializes global variables in the graph.
for step in range(2001):  # 2000회 반복
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})  # cost, h, train 계산
    if step % 10 == 0:  # 10회 반복 마다
        print(step, "Cost:", cost_val, "\nPrediction:\n", hy_val)  # step, cost, h 출력

# 학습 출력 train output
'''
0 Cost: 21027.0 
Prediction:
 [[22.048063 ]
 [21.619772 ]
 ...
 [31.36112  ]
 [24.986364 ]]
10 Cost: 95.976326 
Prediction:
 [[157.11063 ]
 [183.99283 ]
 ...
 [167.48862 ]
 [193.25117 ]]
 1990 Cost: 24.863274 
Prediction:
 [[154.4393  ]
 [185.5584  ]
 ...
 [158.27443 ]
 [192.79778 ]]
2000 Cost: 24.722485 
Prediction:
 [[154.42894 ]
 [185.5586  ]
 ...
 [158.24257 ]
 [192.79166 ]]
'''

# 점수 질의 Ask my score
print("Your score will be ", sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))  # [100, 70, 101] 일때, h 값
print("Other scores will be ", sess.run(hypothesis, feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))  # [[60, 70, 110], [90, 100, 80]] 일때, h 값
'''
Your score will be  [[ 181.73277283]]
Other scores will be  [[ 145.86265564]
 [ 187.23129272]]

'''

'''
실행결과
Your score will be  [[181.73277]]
Other scores will be  [[145.86266]
 [187.2313 ]]
'''
