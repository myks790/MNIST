import tensorflow as tf
import matplotlib.pyplot as plt

if __name__ == '__main__':
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # 첫번째 이미지 확인 방법
    # plt.imshow(x_train[0], cmap='Greys', interpolation="nearest")
    # plt.show()

    # 모델 층 쌓기
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # 모델의 옵티마이저와 손실함수 선택
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 모델 훈련
    model.fit(x_train, y_train, epochs=5)

    # 모델 평가
    model.evaluate(x_test, y_test, verbose=2)
