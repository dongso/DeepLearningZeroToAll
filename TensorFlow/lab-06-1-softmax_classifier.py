# Lab 6 소프트맥스 분류기 Softmax Classifier
import tensorflow as tf  # tensorflow
tf.set_random_seed(777)  # 랜덤 시드 설정 for reproducibility
x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5], [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]]  # X 데이터
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]  # Y 데이터
X = tf.placeholder("float", [None, 4])  # X 플레이스홀더
Y = tf.placeholder("float", [None, 3])  # Y 플레이스홀더
nb_classes = 3  # 3종 분류
W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')  # W 변수
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')  # b 변수
# tf.nn.softmax 로 소프트맥스 활성화 함수를 계산한다. tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)  # 가설 h
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))  # 크로스 엔트로피 비용/손실 함수 Cross entropy cost/loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)  # 경사하강법 비용 최소화
with tf.Session() as sess:  # 그래프 실행 Launch graph
    sess.run(tf.global_variables_initializer())  # 변수 초기화
    for step in range(2001):  # 2000회 반복
            _, cost_val = sess.run([optimizer, cost], feed_dict={X: x_data, Y: y_data})  # optimizer, cost 실행
            if step % 200 == 0:  # 200회 마다 
                print(step, cost_val)  # step, cost 출력
    print('--------------')
    # 테스트와 원핫 인코딩 Testing & One-hot encoding
    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})  # X 데이터 입력한 h 결과
    print(a, sess.run(tf.argmax(a, 1)))  # 원핫 인코딩 결과를 argmax로 최대값 번호 출력

    print('--------------')
    b = sess.run(hypothesis, feed_dict={X: [[1, 3, 4, 3]]})  # X 데이터 입력한 h 결과
    print(b, sess.run(tf.argmax(b, 1)))  # 원핫 인코딩 결과를 argmax로 최대값 번호 출력

    print('--------------')
    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0, 1]]})  # X 데이터 입력한 h 결과
    print(c, sess.run(tf.argmax(c, 1)))  # 원핫 인코딩 결과를 argmax로 최대값 번호 출력

    print('--------------')
    all = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]})  # X 데이터 입력한 h 결과
    print(all, sess.run(tf.argmax(all, 1)))  # 원핫 인코딩 결과를 argmax로 최대값 번호 출력

'''
0 6.926112
200 0.6005015
400 0.47295815
600 0.37342924
800 0.28018373
1000 0.23280522
1200 0.21065344
1400 0.19229904
1600 0.17682323
1800 0.16359556
2000 0.15216158
-------------
[[1.3890490e-03 9.9860185e-01 9.0613084e-06]] [1]
-------------
[[0.9311919  0.06290216 0.00590591]] [0]
-------------
[[1.2732815e-08 3.3411323e-04 9.9966586e-01]] [2]
-------------
[[1.3890490e-03 9.9860185e-01 9.0613084e-06]
 [9.3119192e-01 6.2902197e-02 5.9059085e-03]
 [1.2732815e-08 3.3411323e-04 9.9966586e-01]] [1 0 2]
'''


'''
실행결과
1600 0.17682323
1800 0.16359556
2000 0.15216158
--------------
[[1.3890490e-03 9.9860185e-01 9.0613084e-06]] [1]
--------------
[[0.9311919  0.06290216 0.00590591]] [0]
--------------
[[1.2732815e-08 3.3411323e-04 9.9966586e-01]] [2]
--------------
[[1.3890490e-03 9.9860185e-01 9.0613084e-06]
 [9.3119192e-01 6.2902160e-02 5.9059118e-03]
 [1.2732815e-08 3.3411323e-04 9.9966586e-01]] [1 0 2]
'''
