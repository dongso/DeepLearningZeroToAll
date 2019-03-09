# Lab 4 다중 변수 선형 회귀 Multi-variable linear regression
import tensorflow as tf  # tensorflow
tf.set_random_seed(777)  # 랜덤 시드 설정 for reproducibility
x_data = [[73., 80., 75.], [93., 88., 93.], [89., 91., 90.], [96., 98., 100.], [73., 66., 70.]]  # x 데이터
y_data = [[152.], [185.], [180.], [196.], [142.]]  # y 데이터
# feed_dict 로 입력하게될 플레이스홀더 placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])  # X 플레이스홀더
Y = tf.placeholder(tf.float32, shape=[None, 1])  # Y 플레이스홀더
W = tf.Variable(tf.random_normal([3, 1]), name='weight')  # W 변수
b = tf.Variable(tf.random_normal([1]), name='bias')  # b 변수
hypothesis = tf.matmul(X, W) + b  # 가설 Hypothesis
cost = tf.reduce_mean(tf.square(hypothesis - Y))  # 비용/손실 함수 정의 Simplified cost/loss function
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)  # 최소화를 위한 옵티마이저 Minimize
train = optimizer.minimize(cost)  # 비용 함수 최소화
sess = tf.Session()  # 세션에서 그래프 실행 Launch the graph in a session.
sess.run(tf.global_variables_initializer())  # 그래프에서 변수 초기화 Initializes global variables in the graph.
for step in range(2001):  # 2000회 반복
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})  # cost, h, train 계산
    if step % 10 == 0:  # 10회 반복 마다
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)  # step, cost, h 출력
'''
0 Cost:  7105.46
Prediction:
 [[ 80.82241058]
 [ 92.26364136]
 [ 93.70250702]
 [ 98.09217834]
 [ 72.51759338]]
10 Cost:  5.89726
Prediction:
 [[ 155.35159302]
 [ 181.85691833]
 [ 181.97254944]
 [ 194.21760559]
 [ 140.85707092]]

...

1990 Cost:  3.18588
Prediction:
 [[ 154.36352539]
 [ 182.94833374]
 [ 181.85189819]
 [ 194.35585022]
 [ 142.03240967]]
2000 Cost:  3.1781
Prediction:
 [[ 154.35881042]
 [ 182.95147705]
 [ 181.85035706]
 [ 194.35533142]
 [ 142.036026  ]]

'''

'''
실행결과
1980 Cost:  3.1944592 
Prediction:
 [[154.36868]
 [182.94485]
 [181.85355]
 [194.35635]
 [142.02844]]
1990 Cost:  3.1866612 
Prediction:
 [[154.36398]
 [182.94801]
 [181.85204]
 [194.35587]
 [142.03204]]
2000 Cost:  3.178877 
Prediction:
 [[154.3593 ]
 [182.95117]
 [181.85052]
 [194.35541]
 [142.03566]]
'''