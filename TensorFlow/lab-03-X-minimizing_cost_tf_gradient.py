# Lab 3 비용 최소화 Minimizing Cost
# 선택적으로 진행 할 것. This is optional
import tensorflow as tf  # tensorflow
# 그래프 입력 tf Graph Input
X = [1, 2, 3]  # X 데이터 
Y = [1, 2, 3]  # Y 데이터
W = tf.Variable(5.)  # 잘못된 가중치 설정 Set wrong model weights
hypothesis = X * W  # 리니어 모델 Linear model
gradient = tf.reduce_mean((W * X - Y) * X) * 2  # 직접 경사값 계산 Manual gradient
cost = tf.reduce_mean(tf.square(hypothesis - Y))  # 비용/손실 함수 정의 cost/loss function
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)  # 경사 하강 옵티마이저로 최소화 Minimize: Gradient Descent Optimizer
gvs = optimizer.compute_gradients(cost)  # 옵티마이저로부터 계산된 경사값 얻기 Get gradients
# 만약 필요하다면 경사값을 수정할 수 있다. Optional: modify gradient if necessary
# gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
apply_gradients = optimizer.apply_gradients(gvs)  # (수정한) 경사값을 적용한다. Apply gradients
with tf.Session() as sess:  # 세션에서 그래프를 실행한다. Launch the graph in a session.
    sess.run(tf.global_variables_initializer())  # 그래프에서 변수를 초기화한다. Initializes global variables in the graph.
    for step in range(101):  # 100회 반복
        gradient_val, gvs_val, _ = sess.run([gradient, gvs, apply_gradients])  # gradient, gvs, apply_gradients에 대한 작업 실행
        print(step, gradient_val, gvs_val)  # step, gradient_val, gvs_val 출력
'''
0 37.333332 [(37.333336, 5.0)]
1 33.84889 [(33.84889, 4.6266665)]
2 30.689657 [(30.689657, 4.2881775)]
3 27.825289 [(27.825289, 3.981281)]
...
97 0.0027837753 [(0.0027837753, 1.0002983)]
98 0.0025234222 [(0.0025234222, 1.0002704)]
99 0.0022875469 [(0.0022875469, 1.0002451)]
100 0.0020739238 [(0.0020739238, 1.0002222)]
'''
'''
실행결과
0 37.333332 [(37.333336, 4.6266665)]
1 33.84889 [(33.84889, 4.2881775)]
2 30.689657 [(30.689657, 3.9812808)]
...
98 0.0025234222 [(0.0025234222, 1.0002451)]
99 0.0022875469 [(0.0022875469, 1.0002222)]
100 0.0020739238 [(0.0020739238, 1.0002015)]
'''