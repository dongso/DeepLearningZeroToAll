# Lab 4 다중 변수 선형 회귀 Multi-variable linear regression
import tensorflow as tf  # tensorflow
tf.set_random_seed(777)  # 랜덤 시드 설정 for reproducibility
x1_data = [73., 93., 89., 96., 73.]  # x1 데이터
x2_data = [80., 88., 91., 98., 66.]  # x2 데이터
x3_data = [75., 93., 90., 100., 70.]  # x3 데이터
y_data = [152., 185., 180., 196., 142.]  # y 데이터
# feed_dict 하게될 텐서를 위한 플레이스홀더 placeholders for a tensor that will be always fed.
x1 = tf.placeholder(tf.float32)  # x1 플레이스홀더
x2 = tf.placeholder(tf.float32)  # x2 플레이스홀더
x3 = tf.placeholder(tf.float32)  # x3 플레이스홀더
Y = tf.placeholder(tf.float32)  # Y 플레이스홀더
w1 = tf.Variable(tf.random_normal([1]), name='weight1')  # weight1 이름의 변수 w1.
w2 = tf.Variable(tf.random_normal([1]), name='weight2')  # weight2 이름의 변수 w2.
w3 = tf.Variable(tf.random_normal([1]), name='weight3')  # weight3 이름의 변수 w3.
b = tf.Variable(tf.random_normal([1]), name='bias')  # bias 이름의 변수 b.
hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b  # 가설 h 정의
cost = tf.reduce_mean(tf.square(hypothesis - Y))  # 비용/손실 함수 정의 cost/loss function
# 경사하강 옵티마이저로 최소화. Minimize. Need a very small learning rate for this data set
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)  # 데이터셋에 대한 매우 작은 학습율 설정.
train = optimizer.minimize(cost)  # 옵티마이저로 비용함수 최소화
sess = tf.Session()  # 세션에서 그래프 실행 Launch the graph in a session.
sess.run(tf.global_variables_initializer())  # 그래프에서 변수 초기화 Initializes global variables in the graph.
for step in range(2001):  # 2000회 반복
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data}) # x1, x2, x3, y 데이터 입력하여 cost, h, train 계산
    if step % 10 == 0:  # step 10회 마다
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)  # step, cost, h 값 출력
'''
0 Cost:  19614.8
Prediction:
 [ 21.69748688  39.10213089  31.82624626  35.14236832  32.55316544]
10 Cost:  14.0682
Prediction:
 [ 145.56100464  187.94958496  178.50236511  194.86721802  146.08096313]

 ...

1990 Cost:  4.9197
Prediction:
 [ 148.15084839  186.88632202  179.6293335   195.81796265  144.46044922]
2000 Cost:  4.89449
Prediction:
 [ 148.15931702  186.8805542   179.63194275  195.81971741  144.45298767]
'''

'''
실행결과
1980 Cost:  4.9475894 
Prediction:
 [148.14153 186.8927  179.62645 195.81604 144.46869]
1990 Cost:  4.9222474 
Prediction:
 [148.15    186.8869  179.62906 195.81778 144.4612 ]
2000 Cost:  4.8970113 
Prediction:
 [148.15845 186.8811  179.63167 195.81953 144.45372]
'''
