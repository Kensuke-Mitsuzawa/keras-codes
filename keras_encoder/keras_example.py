#! -*- coding: utf-8 -*-
# keras core
from keras.layers import LSTM
from keras.models import Sequential
from keras import backend as K  # to avoid error inside keras
from keras.callbacks import EarlyStopping
from keras.models import load_model
# data object
import numpy as np

def lstm_short_example():
    """kerasのlstmの使い方チュートリアル的な"""
    LSTM_size_1 = 5
    ## word embeddingを使いたい場合はここを変更すればOK?3階のテンソル
    data = [
        [[ 1 ,3 ],[ 1 ,3 ],[0, 0]],
        [[ 2 ,4 ],[ 2 ,4 ],[0, 0]],
        [[ 3 ,5 ],[ 3 ,5 ],[0, 0]],
        [[ 1 ,3 ],[ 1 ,2 ],[0, 0]],
        [[ 1 ,3 ],[ 1 ,2 ],[1,3]],
    ]

    data_dim = 2
    timesteps = 3

    ## make encoder
    data = np.array(data)

    ## モデルコアの部分は独立させても良い。auto encoderに必要な要素を加える前に .load_weights()を実行すれば良い
    # input_shapeは(1文の最大単語数,1単語を表現する次元数)
    encoder = Sequential([LSTM(units=10, input_shape=(3,2), activation='tanh', return_sequences=True)], name='encoder')
    decoder = Sequential([LSTM(output_dim=2, input_dim=10, activation='tanh', return_sequences=True)], name='decoder')

    autoencoder = Sequential()
    autoencoder.add(encoder)
    autoencoder.add(decoder)
    autoencoder.compile(loss='mse', optimizer='RMSprop')
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    training_log = autoencoder.fit(x=data, y=data, epochs=2, shuffle=True, validation_split=0.2)
    print(training_log.history)

    ## converting into vector
    test_data = data = np.array([
        [[ 1 ,3 ],[ 1 ,3 ],[0,0]],
        [[ 2 ,4 ],[ 2 ,4 ],[0,0]],
        [[ 3 ,5 ],[ 3 ,5 ],[0,0]],
        [[ 1 ,3 ],[ 1 ,2 ],[0,0]],
        [[100, 1], [500, 100], [0,0]]
    ])
    encoded_vector = encoder.predict(x=test_data)
    from scipy import spatial
    print(spatial.distance.cosine(encoded_vector[0][-1], encoded_vector[3][-1]))
    print(spatial.distance.cosine(encoded_vector[0][-1], encoded_vector[4][-1]))

    K.clear_session()  # to avoid error inside keras

if __name__ == '__main__':
    lstm_short_example()