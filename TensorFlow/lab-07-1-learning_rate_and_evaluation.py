# Lab 7 학습율과 평가 Learning rate and Evaluation
import tensorflow as tf  # tensorflow
tf.set_random_seed(777)  # 랜덤 시드 설정 for reproducibility
x_data = [[1, 2, 1], [1, 3, 2], [1, 3, 4], [1, 5, 5], [1, 7, 5], [1, 2, 5], [1, 6, 6], [1, 7, 7]]  # X 데이터
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]  # Y 데이터
# 이 테스트 데이터셋으로 모델 평가 Evaluation our model using this test dataset
x_test = [[2, 1, 1], [3, 1, 2], [3, 3, 4]]  # X 테스트 데이터 
y_test = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]  # Y 테스트 데이터
X = tf.placeholder("float", [None, 3])  # X 플레이스홀더
Y = tf.placeholder("float", [None, 3])  # Y 플레이스홀더
W = tf.Variable(tf.random_normal([3, 3]))  # W 변수
b = tf.Variable(tf.random_normal([3]))  # b 변수
# 소프트맥스 계산 tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))  # 크로스 엔트로피 비용/손실 함수 Cross entropy cost/loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)  # 학습율을 작은 값으로 설정 Try to change learning_rate to small numbers
# 예측한 값이 올바른지 모델 평가 Correct prediction Test model
prediction = tf.argmax(hypothesis, 1)
is_correct = tf.equal(prediction, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
with tf.Session() as sess:  # 그래프 실행 Launch graph
    sess.run(tf.global_variables_initializer())  # 변수 초기화 Initialize TensorFlow variables
    for step in range(201):  # 200회 반복
        cost_val, W_val, _ = sess.run([cost, W, optimizer], feed_dict={X: x_data, Y: y_data})  # cost, W
        print(step, cost_val, W_val)  # step, cost, W 출력
    print("Prediction:", sess.run(prediction, feed_dict={X: x_test}))  # 예측 predict
    print("Accuracy: ", sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))  # 정확도 계산 Calculate the accuracy

'''
when lr = 1.5

0 5.73203 [[-0.30548954  1.22985029 -0.66033536]
 [-4.39069986  2.29670858  2.99386835]
 [-3.34510708  2.09743214 -0.80419564]]
1 23.1494 [[ 0.06951046  0.29449689 -0.0999819 ]
 [-1.95319986 -1.63627958  4.48935604]
 [-0.90760708 -1.65020132  0.50593793]]
2 27.2798 [[ 0.44451016  0.85699677 -1.03748143]
 [ 0.48429942  0.98872018 -0.57314301]
 [ 1.52989244  1.16229868 -4.74406147]]
3 8.668 [[ 0.12396193  0.61504567 -0.47498202]
 [ 0.22003263 -0.2470119   0.9268558 ]
 [ 0.96035379  0.41933775 -3.43156195]]
4 5.77111 [[-0.9524312   1.13037777  0.08607888]
 [-3.78651619  2.26245379  2.42393875]
 [-3.07170963  3.14037919 -2.12054014]]
5 inf [[ nan  nan  nan]
 [ nan  nan  nan]
 [ nan  nan  nan]]
6 nan [[ nan  nan  nan]
 [ nan  nan  nan]
 [ nan  nan  nan]]
 ...
Prediction: [0 0 0]
Accuracy:  0.0
-------------------------------------------------
When lr = 1e-10

0 5.73203 [[ 0.80269563  0.67861295 -1.21728313]
 [-0.3051686  -0.3032113   1.50825703]
 [ 0.75722361 -0.7008909  -2.10820389]]
1 5.73203 [[ 0.80269563  0.67861295 -1.21728313]
 [-0.3051686  -0.3032113   1.50825703]
 [ 0.75722361 -0.7008909  -2.10820389]]
...
199 5.73203 [[ 0.80269563  0.67861295 -1.21728313]
 [-0.3051686  -0.3032113   1.50825703]
 [ 0.75722361 -0.7008909  -2.10820389]]
200 5.73203 [[ 0.80269563  0.67861295 -1.21728313]
 [-0.3051686  -0.3032113   1.50825703]
 [ 0.75722361 -0.7008909  -2.10820389]]
Prediction: [0 0 0]
Accuracy:  0.0
-------------------------------------------------
When lr = 0.1

0 5.73203 [[ 0.72881663  0.71536207 -1.18015325]
 [-0.57753736 -0.12988332  1.60729778]
 [ 0.48373488 -0.51433605 -2.02127004]]
1 3.318 [[ 0.66219079  0.74796319 -1.14612854]
 [-0.81948912  0.03000021  1.68936598]
 [ 0.23214608 -0.33772916 -1.94628811]]
...
199 0.672261 [[-1.15377033  0.28146935  1.13632679]
 [ 0.37484586  0.18958236  0.33544877]
 [-0.35609841 -0.43973011 -1.25604188]]
200 0.670909 [[-1.15885413  0.28058422  1.14229572]
 [ 0.37609792  0.19073224  0.33304682]
 [-0.35536593 -0.44033223 -1.2561723 ]]
Prediction: [2 2 2]
Accuracy:  1.0
'''

'''
실행결과
199 0.67226064 [[-1.148668    0.28236815  1.1303252 ]
 [ 0.37357444  0.18841408  0.33788887]
 [-0.35681716 -0.43911377 -1.2559391 ]]
200 0.67090875 [[-1.1537704   0.28146926  1.1363264 ]
 [ 0.37484607  0.1895825   0.3354488 ]
 [-0.35609835 -0.43973002 -1.2560416 ]]
Prediction: [2 2 2]
Accuracy:  1.0
'''