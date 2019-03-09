# Lab 5 로지스틱 회귀 분류기 Logistic Regression Classifier
import tensorflow as tf  # tensorflow
tf.set_random_seed(777)  # 랜덤 시드 설정 for reproducibility
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]  # X 데이터
y_data = [[0], [0], [0], [1], [1], [1]]  # Y 데이터
# feed_dict로 입력하게될 텐서 플레이스홀더 placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 2])  # X 플레이스홀더
Y = tf.placeholder(tf.float32, shape=[None, 1])  # Y 플레이스홀더
W = tf.Variable(tf.random_normal([2, 1]), name='weight')  # W 변수
b = tf.Variable(tf.random_normal([1]), name='bias')  # b 변수
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)  # 시그모이드 함수를 사용한 가설 정의 Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
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
0 1.73078
200 0.571512
400 0.507414
600 0.471824
800 0.447585
...
9200 0.159066
9400 0.15656
9600 0.154132
9800 0.151778
10000 0.149496

Hypothesis:  [[ 0.03074029]
 [ 0.15884677]
 [ 0.30486736]
 [ 0.78138196]
 [ 0.93957496]
 [ 0.98016882]]
Correct (Y):  [[ 0.]
 [ 0.]
 [ 0.]
 [ 1.]
 [ 1.]
 [ 1.]]
Accuracy:  1.0
'''

'''
실행결과
9600 0.15413196
9800 0.15177834
10000 0.1494956

Hypothesis:  [[0.03074028]
 [0.15884677]
 [0.30486736]
 [0.7813819 ]
 [0.93957496]
 [0.9801688 ]] 
Correct (Y):  [[0.]
 [0.]
 [0.]
 [1.]
 [1.]
 [1.]] 
Accuracy:  1.0
'''