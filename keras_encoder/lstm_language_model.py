#! -*- coding: utf-8 -*-
# keras core
from keras.models import Sequential, Model
from keras.layers.core import Dropout, Activation
from keras.layers import LSTM, TimeDistributed, Dense
from keras import backend as K  # to avoid error inside keras
from keras.callbacks import EarlyStopping
from keras.models import load_model
# gensim
try:
    from gensim.models import KeyedVectors, Word2Vec
except:
    from gensim.models import Word2Vec
    KeyedVectors = None
# numpy
import numpy as np
# data object
from keras_encoder.utils import InputTextObject
# tying
from typing import List, Dict, Tuple, Any, Union, Callable, TypeVar
# logger
from keras_encoder.logger import logger
# else
from sklearn.preprocessing import normalize
import os
import traceback



"""* Summary
LSTMを利用した文の「もっともらしさ」を計算するコード


* Models

- LstmLanguageModel1: もっとも単純な言語モデル
"""


class LanguageScoreObject(object):
    """encodingのステップで利用するオブジェクト
    1テキストの情報を保持する
    """
    def __init__(self, text_id:int, score:float, **args):
        self.text_id = text_id
        self.score = score
        self.args = args


class LstmLanguageModelGenerator1(object):
    """LSTMを利用した言語モデルを作成する

    * Model
    - この言語モデルは「入力文の次の単語を予測する」LSTMモデル
    """
    def __init__(self,
                 word_embedding:Union[Word2Vec, KeyedVectors],
                 limit_diff_max_min:int=20.0,
                 hidden_unit:int=200,
                 max_word_length:int=15,
                 epoch:int=100,
                 validation_ratio:float=0.2,
                 activation:str='sigmoid',
                 loss_function:str='binary_crossentropy',
                 optimizer:str='adam',
                 is_normalize:bool = False):
        """* Parameters
        - word_embedding: gensim実装のword2vecモデル
        - hidden_unit: lstmの隠れユニット数
        - max_word_length: 1文の最大単語数。-1を指定すると、自動的に入力中で最大の単語数を検出する
        - activation: lstmのactivation関数
        - epoch: 学習のepoch回数。
        - validation_ratio: 入力データ中でvalidation dataに使う割合
        - is_normalize: 入力ベクトルをL2ノルム正規化する。勾配爆発して学習時にlossがnanになることを防ぐため。
        - optimizer: 最適化関数"RMSprop"か"adam"あたりがオススメ
        - loss_function: 目的関数. "mse"が最善
        - limit_diff_max_min: 稀にデータ中に変なデータが発生して勾配がが学習できずinfになってしまうことがある。それを防ぐための上限値
        """
        # word embeddingのオブジェクトはこの時点で受け取っておく
        self.hidden_layer = hidden_unit
        self.max_word_length = max_word_length
        self.word_embedding = word_embedding
        self.is_normalize = is_normalize
        embedding_matrix_size = self.word_embedding.syn0.shape
        self.vocabulary = embedding_matrix_size[0]
        self.dimension_word_embedding = embedding_matrix_size[1]
        self.activation = activation
        self.epoch = epoch
        self.validation_ratio = validation_ratio
        self.loss_function = loss_function
        self.optimizer = optimizer
        ## 稀にデータ中に変なデータが発生して勾配がが学習できずinfになってしまうことがある。それを防ぐための上限値 ##
        self.limit_diff_max_min = limit_diff_max_min

    def __del__(self):
        K.clear_session()  # to avoid error inside keras

    def __detect_max_word_length(self, seq_input_text_object:List[InputTextObject]):
        """* What you can do
        - 最大の単語数を探索する
        """
        max_word_length = 0
        for text_obj in seq_input_text_object:
            if len(text_obj.seq_token) > max_word_length:
                max_word_length = len(text_obj.seq_token)
        return max_word_length

    def offset_seq(self, seq:List[str])->Tuple[List[str],List[str]]:
        """* What you can do
        - 入力リストから2つのリストを生成する
        - リスト１とリスト２はインデックスが１つずれているだけ。
            - リスト１は1番めのインデックスから末尾なしまで
            - リスト２は２番めのインデックスから末尾まで
        """
        return seq[:-1], seq[1:]

    def pad_vector(self, seq_word_vector_array:List[np.ndarray])->np.ndarray:
        """* What you can do
        - 入力データ([単語のword embeddingベクトル])にpadding処理を実施する

        * Output
        - 行列; (設定単語数*word embedding次元数)
        """
        if len(seq_word_vector_array) < self.max_word_length:
            ## left-paddingであることに注意 ##
            pad_seq_word_vector_array = ([np.zeros(shape=self.dimension_word_embedding)] * (
            self.max_word_length - len(seq_word_vector_array))) + seq_word_vector_array
        else:
            pad_seq_word_vector_array = seq_word_vector_array[:self.max_word_length]

        pad_seq_word_embedding_matrix = np.array(pad_seq_word_vector_array)
        return pad_seq_word_embedding_matrix

    def generate_input_tensor_training(self, seq_input_text_object:List[InputTextObject])->Tuple[np.ndarray, np.ndarray]:
        """* What you can do
        - 入力単語からモデルの入力tensorを作成する
        - word embeddingを利用して、単語を分散表現に変換する

        * Return
        - ndarrayで表現される3階のtensor
            - (サンプル文数, 1文の最大単語数, 単語を表現する次元数)
        - 1文の最大単語より単語数が少ない文は0でpaddingされる
        """
        if self.max_word_length == -1:
            self.max_word_length = self.__detect_max_word_length(seq_input_text_object=seq_input_text_object)

        seq_sentence_matrix_x = [None] * len(seq_input_text_object)
        seq_sentence_matrix_y = [None] * len(seq_input_text_object)

        for i, text_obj in enumerate(sorted(seq_input_text_object, key=lambda text_obj: text_obj.text_id)):
            seq_token_x, seq_token_y = self.offset_seq(seq=text_obj.seq_token)
            ## word vectorに変換する ##
            seq_word_vector_array_x = [self.word_embedding[word] for word in seq_token_x if word in self.word_embedding]
            seq_word_vector_array_y = [self.word_embedding[word] for word in seq_token_y if word in self.word_embedding]
            if len(seq_word_vector_array_x) == 0:
                logger.warning('Skip. No word embedding for {}'.format(text_obj.text))
                continue
            ## 単語のpadding処理を実行する ##
            pad_seq_word_embedding_matrix_x = self.pad_vector(seq_word_vector_array_x)
            pad_seq_word_embedding_matrix_y = self.pad_vector(seq_word_vector_array_y)

            ## nan, inifiniteのチェックを実施する ##
            if np.any(np.isnan(pad_seq_word_embedding_matrix_x)) or np.any(np.isnan(pad_seq_word_embedding_matrix_y)):
                logger.warning('Skip text-id={} because it has nan value'.format(text_obj.text_id))
                continue
            if not np.all(np.isfinite(pad_seq_word_embedding_matrix_x)) or not np.all(np.isfinite(pad_seq_word_embedding_matrix_y)):
                logger.warning('Skip text-id={} because it has infinite value'.format(text_obj.text_id))
                continue
            if self.is_normalize:
                ## ベクトルの正規化を実施する。サンプルごとに独立に正規化する ##
                pad_seq_word_embedding_matrix_x = normalize(X=pad_seq_word_embedding_matrix_x, axis=1)
                pad_seq_word_embedding_matrix_y = normalize(X=pad_seq_word_embedding_matrix_y, axis=1)

            ## 値の最大と最小チェックを実施する。差分が大きすぎる場合は異常、とみなして、データからはじく
            diff_max_min_matrix_x = abs(pad_seq_word_embedding_matrix_x.min()) + abs(pad_seq_word_embedding_matrix_x.max())
            diff_max_min_matrix_y = abs(pad_seq_word_embedding_matrix_y.min()) + abs(pad_seq_word_embedding_matrix_y.max())
            if diff_max_min_matrix_x > self.limit_diff_max_min or diff_max_min_matrix_y > self.limit_diff_max_min:
                logger.warning('Skip text-id={} because it has large gap in min value and max value. gap={}, gap_limit={}'.format(text_obj.text_id, diff_max_min_matrix_x, self.limit_diff_max_min))
                continue
            seq_sentence_matrix_x[i] = pad_seq_word_embedding_matrix_x
            seq_sentence_matrix_y[i] = pad_seq_word_embedding_matrix_y

        seq_sentence_matrix_x = [matrix for matrix in seq_sentence_matrix_x if not matrix is None]
        seq_sentence_matrix_y = [matrix for matrix in seq_sentence_matrix_y if not matrix is None]

        return np.array(seq_sentence_matrix_x), np.array(seq_sentence_matrix_y)

    def generate_input_tensor_test(self, seq_input_text_object:List[InputTextObject])->np.ndarray:
        """* What you can do
        - 入力単語からモデルの入力tensorを作成する
        - word embeddingを利用して、単語を分散表現に変換する

        * Return
        - ndarrayで表現される3階のtensor
            - (サンプル文数, 1文の最大単語数, 単語を表現する次元数)
        - 1文の最大単語より単語数が少ない文は0でpaddingされる
        """
        if self.max_word_length == -1:
            self.max_word_length = self.__detect_max_word_length(seq_input_text_object=seq_input_text_object)

        seq_sentence_matrix_x = [None] * len(seq_input_text_object)

        for i, text_obj in enumerate(sorted(seq_input_text_object, key=lambda text_obj: text_obj.text_id)):
            seq_token_x, seq_token_y = self.offset_seq(seq=text_obj.seq_token)
            ## word vectorに変換する ##
            seq_word_vector_array_x = [self.word_embedding[word] for word in seq_token_x if word in self.word_embedding]
            if len(seq_word_vector_array_x) == 0:
                logger.warning('Skip. No word embedding for {}'.format(text_obj.text))
                continue
            ## 単語のpadding処理を実行する ##
            pad_seq_word_embedding_matrix_x = self.pad_vector(seq_word_vector_array_x)

            ## nan, inifiniteのチェックを実施する ##
            if np.any(np.isnan(pad_seq_word_embedding_matrix_x)):
                logger.warning('Skip text-id={} because it has nan value'.format(text_obj.text_id))
                continue
            if not np.all(np.isfinite(pad_seq_word_embedding_matrix_x)):
                logger.warning('Skip text-id={} because it has infinite value'.format(text_obj.text_id))
                continue
            if self.is_normalize:
                ## ベクトルの正規化を実施する。サンプルごとに独立に正規化する ##
                pad_seq_word_embedding_matrix_x = normalize(X=pad_seq_word_embedding_matrix_x, axis=1)

            ## 値の最大と最小チェックを実施する。差分が大きすぎる場合は異常、とみなして、データからはじく
            diff_max_min_matrix_x = abs(pad_seq_word_embedding_matrix_x.min()) + abs(pad_seq_word_embedding_matrix_x.max())
            if diff_max_min_matrix_x > self.limit_diff_max_min:
                logger.warning('Skip text-id={} because it has large gap in min value and max value. gap={}, gap_limit={}'.format(text_obj.text_id, diff_max_min_matrix_x, self.limit_diff_max_min))
                continue
            seq_sentence_matrix_x[i] = pad_seq_word_embedding_matrix_x

        seq_sentence_matrix_x = [matrix for matrix in seq_sentence_matrix_x if not matrix is None]

        return np.array(seq_sentence_matrix_x)

    def define_model(self, **args)->Sequential:
        """* What you can do
        - Lstm言語のネットワーク自体を定義する
        """

        language_model = Sequential()
        language_model.add(
            LSTM(units=self.hidden_layer, input_shape=(self.max_word_length, self.dimension_word_embedding),
                 activation=self.activation, return_sequences=True))
        language_model.add(Dropout(0.2))
        language_model.add(TimeDistributed(Dense(units=self.dimension_word_embedding), input_shape=(None, 30, 200)))
        #language_model.add(TimeDistributed(Dense(1), input_shape=(None, 30, 200)))
        language_model.add(Activation('softmax'))

        language_model.compile(loss=self.loss_function, optimizer=self.optimizer)

        return language_model

    def train(self, seq_input_text_object:List[InputTextObject], is_early_stop:bool=True, **args)->Sequential:
        """* What you can do
        - 単純なLstm auto encoderをtrainingする
        """
        input_tensor_x, input_tensor_y = self.generate_input_tensor_training(seq_input_text_object=seq_input_text_object)
        language_model = self.define_model()
        early_stopping = EarlyStopping(monitor='val_loss', patience=2, min_delta=0.01)
        if is_early_stop:
            logger.info(msg='Use early stopping')
            seq_callback = [early_stopping]
        else:
            seq_callback = []

        logger.info(msg='Now training model...')
        training_log = language_model.fit(x=input_tensor_x,
                                          y=input_tensor_y,
                                          epochs=self.epoch,
                                          shuffle=True,
                                          validation_split=self.validation_ratio,
                                          callbacks=seq_callback)
        logger.info(training_log.history)

        self.language_model = language_model

        return language_model

    def save_model(self, filepath:str, trained_language_model:Sequential, **args)->str:
        """* What you can do
        - モデルを保存する
        - この保存方法では、load_model()を実行するだけで、設定もすべて復元される
        """
        if not os.path.exists(os.path.dirname(filepath)):
            raise FileExistsError('No dir name at {}'.format(os.path.dirname(filepath)))
        trained_language_model.save(filepath=filepath, overwrite=True)

        return filepath

    def get_language_score(self,
                           seq_input_text_object:List[InputTextObject],
                           trained_model: Sequential=None,
                           is_sort:bool=True,
                           **args)->List[LanguageScoreObject]:
        """* What you can do
        - 訓練済みモデルを使って、入力の言語モデルスコアを算出する
        - 言語モデルスコアとは「損失関数の合計値」
            - 「訓練済みのモデルで、『入力文の次の単語予測』タスクを解いた時」に、「予測をしくじったスコア」と解釈できる
            - 予測をしくじることが多ければ、それは入力文がおかしい。という解釈になる。

        * Parameters
        - trained_model: 訓練済みの言語モデルLSTMネットワーク。与えていない場合は例外発生
        """
        if not trained_model is None:
            language_model = trained_model
        else:
            if not hasattr(self, 'language_model'):
                raise Exception('This generator has not had trained model yet. You first train model or give existing trained model.')
            language_model = self.language_model

        input_tensor_x, input_tensor_y = self.generate_input_tensor_training(seq_input_text_object=seq_input_text_object)
        seq_language_score_obj = [None] * len(input_tensor_x)
        for vector_index, vector_x in enumerate(input_tensor_x):
            loss_score = language_model.evaluate(x=np.array([vector_x]), y=np.array([input_tensor_y[vector_index]]), verbose=0)
            seq_language_score_obj[vector_index] = LanguageScoreObject(
                text_id=seq_input_text_object[vector_index].text_id,
                score=loss_score,
                text=seq_input_text_object[vector_index].text)
        seq_language_score_obj = [obj for obj in seq_language_score_obj if not obj is None]
        if is_sort:
            seq_language_score_obj = list(sorted(seq_language_score_obj, key=lambda obj: obj.score, reverse=True))

        return seq_language_score_obj

    def load_model(self, path_trained_model, **args)->Sequential:
        """* What you can do
        - 訓練済みのモデル読み込みを実施する
        """
        if not os.path.exists(path_trained_model):
            raise FileExistsError('No trained model at {}'.format(path_trained_model))
        try:
            language_model = load_model(filepath=path_trained_model)
        except:
            logger.error(traceback.format_exc())
            raise Exception()

        return language_model

    @classmethod
    def init_trained_model(cls,
                           word_embedding:Union[Word2Vec, KeyedVectors],
                           path_trained_model:str):
        """* What you can do
        - Generator自体を生成する

        * Note
        - モデルのパラメタは訓練時のパラメタが自動的にセットされる
        """
        language_model_obj = LstmLanguageModelGenerator1(word_embedding=word_embedding)
        cls.language_model = language_model_obj.load_model(path_trained_model)
        input_shape = cls.language_model.input_shape
        language_model_obj.max_word_length = input_shape[1]
        language_model_obj.dimension_word_embedding = input_shape[2]

        return language_model_obj