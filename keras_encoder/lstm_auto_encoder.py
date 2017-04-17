#! -*- coding: utf-8 -*-
# keras core
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM, RepeatVector, Input, normalization
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
# tying
from typing import List, Dict, Tuple, Any, Union, Callable, TypeVar
# logger
from keras_encoder.logger import logger
# psql
from psycopg2.extensions import cursor, connection
# else
from sklearn.preprocessing import normalize
import os
import traceback
import uuid



"""* Summary
LSTMを利用した文のencoderを構築するクラス集


* Models

- LstmEncoderGenerator1: もっとも単純なauto-encoderモデル
    - Training phase: 入力文 -> word embedding -> auto-encoder -> encoder-model
    - Test phase: 入力文 -> word embedding -> encoder-model -> encoded vector
"""


class InputTextObject(object):
    """Training, encodingの両方のステップで利用するオブジェクト
    1テキストの情報を保持する
    """
    def __init__(self, text_id:Union[str,int], text:str, seq_token:List[Union[str,None]], **args):
        self.text_id = text_id
        self.text = text
        self.args = args
        self.seq_token = seq_token


class EncodedVectorObject(object):
    """encodingのステップで利用するオブジェクト
    1テキストの情報を保持する
    """
    def __init__(self, text_id:int, vector:np.ndarray, **args):
        self.text_id = text_id
        self.vector = vector
        self.args = args


class LstmEncoder(object):
    """LSTMを利用したauto encoderを作成する
    そのためのBase class"""
    def define_model(self, **args):
        """* What you can do
        - モデルの構成を設計するメソッド

        * Return
        - モデルのネットワークオブジェクト
        """
        raise NotImplementedError()

    def train(self, **args):
        """* What you can do
        - モデルを訓練するメソッド
        """
        raise NotImplementedError()

    def save_model(self, **args):
        """* What you can do
        - 訓練したモデルを保存する。保存方法は利用するバックエンドに依存する
        """
        raise NotImplementedError()

    def load_model(self, **args):
        """* What you can do
        - 訓練済みのモデルを読み込みする。読み込み方法はバックエンドに依存する
        """

    def encode_text(self, **args):
        """* What you can do
        - 入力文を固定長のベクトルに変換する
        """
        raise NotImplementedError()


class LstmAutoEncoderGenerator1(LstmEncoder):
    """LSTMを利用したauto encoderを作成する
    """
    def __init__(self,
                 word_embedding:Union[Word2Vec, KeyedVectors],
                 limit_diff_max_min:int=20.0,
                 hidden_unit:int=200,
                 max_word_length:int=15,
                 epoch:int=100,
                 validation_ratio:float=0.2,
                 activation:str='tanh',
                 loss_function:str='mse',
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

    def generate_input_tensor(self, seq_input_text_object:List[InputTextObject])->np.ndarray:
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

        seq_sentence_matrix = [None] * len(seq_input_text_object)


        for i, text_obj in enumerate(sorted(seq_input_text_object, key=lambda text_obj: text_obj.text_id)):
            ## word vectorに変換する ##
            seq_word_vector_array = [self.word_embedding[word]
                                     for word in text_obj.seq_token if word in self.word_embedding]
            if len(seq_word_vector_array) == 0:
                logger.warning('Skip. No word embedding for {}'.format(text_obj.text))
                continue
            ## 単語のpadding処理を実行する ##
            if len(seq_word_vector_array) < self.max_word_length:
                ## left-paddingであることに注意 ##
                pad_seq_word_vector_array = ([np.zeros(shape=self.dimension_word_embedding)] * (self.max_word_length - len(seq_word_vector_array))) + seq_word_vector_array
            else:
                pad_seq_word_vector_array = seq_word_vector_array[:self.max_word_length]

            pad_seq_word_embedding_matrix = np.array(pad_seq_word_vector_array)

            ## nan, inifiniteのチェックを実施する ##
            if np.any(np.isnan(pad_seq_word_embedding_matrix)):
                logger.warning('Skip text-id={} because it has nan value'.format(text_obj.text_id))
                continue
            if not np.all(np.isfinite(pad_seq_word_embedding_matrix)):
                logger.warning('Skip text-id={} because it has infinite value'.format(text_obj.text_id))
                continue
            if self.is_normalize:
                ## ベクトルの正規化を実施する。サンプルごとに独立に正規化する ##
                pad_seq_word_embedding_matrix = normalize(X=pad_seq_word_embedding_matrix, axis=1)
            ## 値の最大と最小チェックを実施する。差分が大きすぎる場合は異常、とみなして、データからはじく
            diff_max_min_matrix = abs(pad_seq_word_embedding_matrix.min()) + abs(pad_seq_word_embedding_matrix.max())
            if diff_max_min_matrix > self.limit_diff_max_min:
                logger.warning('Skip text-id={} because it has large gap in min value and max value. gap={}, gap_limit={}'.format(text_obj.text_id, diff_max_min_matrix, self.limit_diff_max_min))
                continue

            seq_sentence_matrix[i] = pad_seq_word_embedding_matrix
        seq_sentence_matrix = [matrix for matrix in seq_sentence_matrix if not matrix is None]
        return np.array(seq_sentence_matrix)

    def define_model(self, **args)->Sequential:
        """* What you can do
        - Lstmのネットワーク自体を定義する
        """

        encoder = Sequential(
            [LSTM(units=self.hidden_layer, input_shape=(self.max_word_length, self.dimension_word_embedding), activation=self.activation, return_sequences=True)],
            name='encoder')
        decoder = Sequential(
            [LSTM(output_dim=self.dimension_word_embedding, input_shape=(self.dimension_word_embedding, self.hidden_layer), activation=self.activation, return_sequences=True)],
            name='decoder')

        autoencoder = Sequential()
        autoencoder.add(encoder)
        autoencoder.add(decoder)
        autoencoder.compile(loss=self.loss_function, optimizer=self.optimizer)

        return autoencoder

    def train(self, seq_input_text_object:List[InputTextObject], is_early_stop:bool=True, **args)->Sequential:
        """* What you can do
        - 単純なLstm auto encoderをtrainingする
        """
        input_tensor = self.generate_input_tensor(seq_input_text_object=seq_input_text_object)
        autoencoder = self.define_model()
        early_stopping = EarlyStopping(monitor='val_loss', patience=2, min_delta=0.01)
        if is_early_stop:
            logger.info(msg='Use early stopping')
            seq_callback = [early_stopping]
        else:
            seq_callback = []

        logger.info(msg='Now training model...')
        training_log = autoencoder.fit(x=input_tensor,
                                       y=input_tensor,
                                       epochs=self.epoch,
                                       shuffle=True,
                                       validation_split=self.validation_ratio,
                                       callbacks=seq_callback)
        logger.info(training_log.history)

        self.auto_encoder = autoencoder

        return autoencoder

    def save_model(self, filepath:str, trained_encoder_model:Sequential, **args)->str:
        """* What you can do
        - モデルを保存する
        - この保存方法では、load_model()を実行するだけで、設定もすべて復元される
        """
        if not os.path.exists(os.path.dirname(filepath)):
            raise FileExistsError('No dir name at {}'.format(os.path.dirname(filepath)))
        trained_encoder_model.save(filepath=filepath, overwrite=True)

        return filepath

    def encode_text(self,
                    seq_input_text_object:List[InputTextObject],
                    trained_model: Sequential=None,
                    encoder_index:int=0,
                    **args)->List[EncodedVectorObject]:
        """* What you can do
        - 訓練済みモデルを使って、入力のベクトル化を実施する

        * Parameters
        - trained_model: 訓練済みのauto-encoder。与えていない場合は例外発生
        - encoder_index: auto-encoderモデルのlayer中で、encoderが存在しているlayerインデックス。通常は最初のインデックス
        """
        if not trained_model is None:
            encoder_model_obj = trained_model
        else:
            if not hasattr(self, 'auto_encoder'):
                raise Exception('This generator has not has trained model yet. You first train model or give existing trained model.')
            encoder_model_obj = self.auto_encoder
        encoder = encoder_model_obj.layers[encoder_index]

        input_tensor = self.generate_input_tensor(seq_input_text_object=seq_input_text_object)
        seq_encoded_tensor = encoder.predict(x=input_tensor)  # type: np.ndarray

        seq_sorted_text_obj = list(sorted(seq_input_text_object, key=lambda text_obj: text_obj.text_id))
        seq_encoded_vector_obj = [None] * len(seq_input_text_object)
        for i, encoded_matrix in enumerate(seq_encoded_tensor):
            ## LSTMに於いては、末尾のvectorが「文の情報の圧縮」である ##
            encoded_text_vector = encoded_matrix[-1]
            seq_encoded_vector_obj[i] = EncodedVectorObject(
                text_id=i,
                vector=encoded_text_vector,
                text=seq_sorted_text_obj[i].text)
        seq_encoded_vector_obj = [obj for obj in seq_encoded_vector_obj if not obj is None]
        return seq_encoded_vector_obj

    def load_model(self, path_trained_model, **args)->Sequential:
        """* What you can do
        - 訓練済みのモデル読み込みを実施する
        """
        if not os.path.exists(path_trained_model):
            raise FileExistsError('No trained model at {}'.format(path_trained_model))
        try:
            auto_encoder = load_model(filepath=path_trained_model)
        except:
            logger.error(traceback.format_exc())
            raise Exception()

        return auto_encoder

    @classmethod
    def init_trained_alddin_encoder(cls,
                                    word_embedding:Union[Word2Vec, KeyedVectors],
                                    path_trained_model:str):
        """* What you can do
        - MulanLstmEncoder自体を生成する

        * Note
        - モデルのパラメタは訓練時のパラメタが自動的にセットされる
        """
        mulan_encoder_obj = LstmAutoEncoderGenerator1(word_embedding=word_embedding)
        cls.auto_encoder = mulan_encoder_obj.load_model(path_trained_model)
        input_shape = cls.auto_encoder.input_shape
        mulan_encoder_obj.max_word_length = input_shape[1]
        mulan_encoder_obj.dimension_word_embedding = input_shape[2]

        return mulan_encoder_obj


BaseEncoder = TypeVar('BaseEncoder', bound=LstmEncoder)
def train_fuman2vec_encoder(encoder_generator:BaseEncoder,
                            path_trained_model:str,
                            psql_post_connection:connection,
                            func_tokenizer:Callable[[str], List[str]],
                            where_clause:str)->BaseEncoder:
    """* What you can do
    - encoderオブジェクトの訓練を実施する
    """
    if not os.path.exists(os.path.dirname(path_trained_model)):
        raise FileExistsError('No directory at {}'.format(os.path.dirname(path_trained_model)))

    post_cur = psql_post_connection.cursor(name=str(uuid.uuid4()))  # type: cursor
    post_cur.itersize = 10000
    sql_base = "SELECT id, text FROM posts " + where_clause
    seq_text_obj = []
    logger.info(msg='Now fetching records....')
    try:
        post_cur.execute(sql_base)
    except:
        logger.error(msg=traceback.format_exc())
        raise Exception()
    else:
        record = post_cur.fetchone()
        while record:
            seq_text_obj.append(
                InputTextObject(text_id=record[0], text=record[1], seq_token=func_tokenizer(record[1])))
            record = post_cur.fetchone()
            if record is None:
                break
        post_cur.close()

    trained_encoder = encoder_generator.train(
        seq_input_text_object=seq_text_obj,
        is_early_stop=True)
    encoder_generator.save_model(filepath=path_trained_model,
                                 trained_encoder_model=trained_encoder)
    psql_post_connection.close()
    return True
