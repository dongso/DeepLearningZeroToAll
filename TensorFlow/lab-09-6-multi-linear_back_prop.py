# http://blog.aloni.org/posts/backprop-with-tensorflow/
# https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b#.b3rvzhx89
# WIP
import tensorflow as tf  # tensorflow
tf.set_random_seed(777)  # 랜덤 시드 설정 reproducibility
# 그래프 입력 tf Graph Input
x_data = [[73., 80., 75.], [93., 88., 93.], [89., 91., 90.], [96., 98., 100.],
          [73., 66., 70.]]  # x 데이터
y_data = [[152.], [185.], [180.], [196.], [142.]]  # y 데이터

# feed_dict 하게될 텐서 플레이스홀더 placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])  # X 플레이스홀더
Y = tf.placeholder(tf.float32, shape=[None, 1])  # Y 플레이스홀더
# 잘못된 가중치 설정 Set wrong model weights
W = tf.Variable(tf.truncated_normal([3, 1]))  # W 변수
b = tf.Variable(5.)  # b 변수
hypothesis = tf.matmul(X, W) + b  # 정방향 전파 Forward prop
print(hypothesis.shape, Y.shape)  # 가설 h shape, Y shape 출력
assert hypothesis.shape.as_list() == Y.shape.as_list()  # shape 차이가 있는지 검사 diff
diff = (hypothesis - Y)  # 차이값 diff
# 체인룰을 적용한 역전파 Back prop (chain rule)
d_l1 = diff
d_b = d_l1
d_w = tf.matmul(tf.transpose(X), d_l1)
print(X, d_l1, d_w)
# 경사값을 사용하여 신경망 업데이트 Updating network using gradients
learning_rate = 1e-6  # 학습율
step = [
    tf.assign(W, W - learning_rate * d_w),
    tf.assign(b, b - learning_rate * tf.reduce_mean(d_b)),
]
# 학습과정을 실행하고 평가한다. 7. Running and testing the training process
RMSE = tf.reduce_mean(tf.square((Y - hypothesis)))  # 오차 제곱 합
sess = tf.InteractiveSession()  # 인터랙티브 세션
init = tf.global_variables_initializer()  # 변수 초기화
sess.run(init)  # 초기화 실행
for i in range(10000):  # 9999회 반복
    print(i, sess.run([step, RMSE], feed_dict={X: x_data, Y: y_data}))  # step, RMSE 계산
print(sess.run(hypothesis, feed_dict={X: x_data}))  # h 계산

'''
실행결과
9997 [[array([[ 1.4688166 ],
       [ 1.0243883 ],
       [-0.52622247]], dtype=float32), 5.001305], 4.1204634]
9998 [[array([[ 1.4688221 ],
       [ 1.0243722 ],
       [-0.52621204]], dtype=float32), 5.001305], 4.1203036]
9999 [[array([[ 1.4688276],
       [ 1.0243561],
       [-0.5262016]], dtype=float32), 5.001305], 4.120116]
[[154.7091 ]
 [182.80887]
 [181.58525]
 [193.7755 ]
 [142.99911]]
'''