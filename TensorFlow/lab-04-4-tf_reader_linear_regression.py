# Lab 4 다중 변수 선형 회귀 Multi-variable linear regression
# https://www.tensorflow.org/programmers_guide/reading_data
import tensorflow as tf  # tensorflow
tf.set_random_seed(777)  # 랜덤 시드 설정 for reproducibility
filename_queue = tf.train.string_input_producer(['data-01-test-score.csv'], shuffle=False, name='filename_queue')  # 콤마 구분 csv 데이터 파일명 큐 생성
reader = tf.TextLineReader()  # 텐서플로우 TestLineReader
key, value = reader.read(filename_queue)  # 파일명 큐로 key, value 읽기
record_defaults = [[0.], [0.], [0.], [0.]]  # 기본값으로 타입 설정한다. Default values, in case of empty columns. Also specifies the type of the decoded result.
xy = tf.decode_csv(value, record_defaults=record_defaults)  # decode_csv 로 xy 값을 읽는다.
train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)  # x batch와 y batch로 구분하여 값을 읽는다. collect batches of csv in
# feed_dict 하게될 텐서 플레이스홀더 placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])  # X 플레이스홀더
Y = tf.placeholder(tf.float32, shape=[None, 1])  # Y 플레이스홀더
W = tf.Variable(tf.random_normal([3, 1]), name='weight')  # W 변수
b = tf.Variable(tf.random_normal([1]), name='bias')  # b 변수
hypothesis = tf.matmul(X, W) + b  # 가설 Hypothesis
cost = tf.reduce_mean(tf.square(hypothesis - Y))  # 비용/손실 함수 Simplified cost/loss function
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)  # 최소화 Minimize
train = optimizer.minimize(cost)  # 옵티마이저로 비용 최소화
sess = tf.Session()  # 세션에서 그래프 실행 Launch the graph in a session.
sess.run(tf.global_variables_initializer())  # 그래프에서 변수 초기화 Initializes global variables in the graph.
# 파일명 큐로 작업 시작 Start populating the filename queue.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
for step in range(2001):  # 2000회 반복
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])  # x y batch 값 얻기
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})  # batch 값을 입력으로 cost, h, train 실행하기
    if step % 10 == 0:  # 10회 마다 step, cost, h 출력
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)
# 파일명 큐 작업 끝내기
coord.request_stop()
coord.join(threads)
# 점수 질의 Ask my score
print("Your score will be ", sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]})) # X 데이터 입력후 h 결과 
print("Other scores will be ", sess.run(hypothesis, feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))  # X 데이터 입력후 h 결과
'''
Your score will be  [[185.33531]]
Other scores will be  [[178.36246]
 [177.03687]]
'''

'''
실행결과
Your score will be  [[185.3353]]
Other scores will be  [[178.36246]
 [177.03687]]
'''