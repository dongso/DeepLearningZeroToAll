# http://blog.aloni.org/posts/backprop-with-tensorflow/
# https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b#.b3rvzhx89
# WIP
import tensorflow as tf  # tensorflow
tf.set_random_seed(777)  # 랜덤 시드 설정 reproducibility
# 그래프 입력 tf Graph Input
x_data = [[1.], [2.], [3.]]  # x 데이터
y_data = [[1.], [2.], [3.]]  # y 데이터
# feed_dict 하게될 텐서 플레이스홀더 placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 1])  # X 플레이스홀더
Y = tf.placeholder(tf.float32, shape=[None, 1])  # Y 플레이스홀더
# 잘못된 값으로 초기화 Set wrong model weights
W = tf.Variable(tf.truncated_normal([1, 1]))
b = tf.Variable(5.)
hypothesis = tf.matmul(X, W) + b  # 정방향 전파 Forward prop
assert hypothesis.shape.as_list() == Y.shape.as_list()  # shape 차이가 있는지 검사 diff
diff = (hypothesis - Y)  # 차이값 diff

# 체인룰을 적용한 역전파 Back prop (chain rule)
d_l1 = diff
d_b = d_l1
d_w = tf.matmul(tf.transpose(X), d_l1)
print(X, W, d_l1, d_w)
# 경사값을 사용하여 신경망 업데이트 Updating network using gradients
learning_rate = 0.1  # 학습율
step = [
    tf.assign(W, W - learning_rate * d_w),
    tf.assign(b, b - learning_rate * tf.reduce_mean(d_b)),
]

# 학습과정을 실행하고 평가한다. 7. Running and testing the training process
RMSE = tf.reduce_mean(tf.square((Y - hypothesis)))  # 오차 제곱 합
sess = tf.InteractiveSession()  # 인터랙티브 세션
init = tf.global_variables_initializer()  # 변수 초기화
sess.run(init)  # 초기화 실행
for i in range(1000):  # 999회 반복
    print(i, sess.run([step, RMSE], feed_dict={X: x_data, Y: y_data}))  # step, RMSE 계산
print(sess.run(hypothesis, feed_dict={X: x_data}))  # h 계산

'''
실행결과
997 [[array([[0.9999972]], dtype=float32), 6.3574607e-06], 5.783818e-12]
998 [[array([[0.99999726]], dtype=float32), 6.2779877e-06], 5.6464464e-12]
999 [[array([[0.9999973]], dtype=float32), 6.1985147e-06], 5.6464464e-12]
[[1.0000035]
 [2.0000007]
 [2.999998 ]]
'''