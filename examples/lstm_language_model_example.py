#! -*- coding: utf-8 -*-
# keras-core
# module
from keras_encoder.lstm_language_model import LstmLanguageModelGenerator1, InputTextObject, LanguageScoreObject
from keras_encoder.utils import preprocess_text
# typing
from typing import List, Tuple, Any, Dict, Callable
# japanese tokenizer
from JapaneseTokenizer import MecabWrapper
# gensim
try:
    from gensim.models import KeyedVectors, Word2Vec
except:
    from gensim.models import Word2Vec
    KeyedVectors = None
# else
import json
import traceback
from functools import partial
import os


def __func_japanese_tokenizer(text:str,
                              tokenizer_obj:MecabWrapper,
                              pos_condition:List[Tuple[str,...]]=None,
                              is_surface:bool=False)->List[str]:
    """* What you can do
    - This is base function tokenizer.
    - You use this function with functools.partial
    """
    if pos_condition is None:
        return tokenizer_obj.tokenize(sentence=text, is_surface=is_surface).convert_list_object()
    else:
        return tokenizer_obj.tokenize(sentence=text, is_surface=is_surface).filter(pos_condition).convert_list_object()


PATH_TRAINING_TEXT = './wikipedia_data/wikipedia-full.json'
PATH_TEST_TEXT = './wikipedia_data/wikipedia-evaluation-full.json'
PATH_ENTITY_VECTOR = './entity_vector/entity_vector.model.bin'
PATH_SAVE_TARINED_MODEL = './trained_language_model.h5'
POS_CONDITION = [('名詞',), ('動詞', '自立'), ('形容詞', '自立'), ('副詞',), ('助動詞',), ('連体詞',)]

## check file existing ##
if not os.path.exists(PATH_TRAINING_TEXT):
    raise FileExistsError()
if not os.path.exists(PATH_ENTITY_VECTOR):
    raise FileExistsError()

## initialize tokenizer funtion ##
tokenizer_obj = MecabWrapper(dictType='neologd')
get_token = partial(__func_japanese_tokenizer,
                    tokenizer_obj=tokenizer_obj,
                    pos_condition=POS_CONDITION,
                    is_surface=False)

## load word embedding ##
try:
    embedding_model = KeyedVectors.load_word2vec_format(PATH_ENTITY_VECTOR,
                                                        **{'binary': True, 'unicode_errors': 'ignore'})
except:
    embedding_model = Word2Vec.load_word2vec_format(PATH_ENTITY_VECTOR,
                                                        **{'binary': True, 'unicode_errors': 'ignore'})


## make training data ##
with open(PATH_TRAINING_TEXT, 'r') as f:
    seq_wikipedia_training_text = json.loads(f.read())

seq_training_input_text_obj = []
for i, wikipedia_article_obj in enumerate(seq_wikipedia_training_text):
    seq_training_input_text_obj += preprocess_text(
        text_index=i,
        func_get_token=get_token,
        dict_wikipedia_article_obj=wikipedia_article_obj)


## make test data ##
with open(PATH_TEST_TEXT, 'r') as f:
    seq_wikipedia_test_text = json.loads(f.read())

seq_test_input_text_obj = []
for i, wikipedia_article_obj in enumerate(seq_wikipedia_test_text):
    seq_test_input_text_obj += preprocess_text(
        text_index=i,
        func_get_token=get_token,
        dict_wikipedia_article_obj=wikipedia_article_obj)

'''
## initialize encoder training-object ##
language_model_generator = LstmLanguageModelGenerator1(
    word_embedding=embedding_model,
    hidden_unit=200,
    max_word_length=30,
    epoch=1,
    validation_ratio=0.2)
## start training ##
trained_encoder_obj = language_model_generator.train(seq_input_text_object=seq_training_input_text_obj, is_early_stop=True)
## save trained-model ##
language_model_generator.save_model(filepath=PATH_SAVE_TARINED_MODEL, trained_language_model=trained_encoder_obj)
del trained_encoder_obj
del language_model_generator'''

tmp_seq_test_input_text_obj = [
    seq_test_input_text_obj[0],
    seq_test_input_text_obj[1],
    InputTextObject(text_id='2-0', text='あああああああいいいうううえええをおお', seq_token=get_token('あああああああいいいうううえええをおお')),
    InputTextObject(text_id='3-0', text='東京の言語は岡本太郎のアフリカで、こいつはディズニー', seq_token=get_token('東京の言語は岡本太郎のアフリカで、こいつはディズニー')),
    InputTextObject(text_id='4-0', text='中国国家統計局は17日、2017年１～３月期の国内総生産（ＧＤＰ）が物価の変動を除いた実質で前年同期比6.9％増えたと発表した。', seq_token=get_token('中国国家統計局は17日、2017年１～３月期の国内総生産（ＧＤＰ）が物価の変動を除いた実質で前年同期比6.9％増えたと発表した。')),
]

## model loading ##
language_model_generator = LstmLanguageModelGenerator1.init_trained_model(
    word_embedding=embedding_model,
    path_trained_model=PATH_SAVE_TARINED_MODEL)
## get language score ##
language_model_generator.get_language_score(seq_input_text_object=tmp_seq_test_input_text_obj)
del language_model_generator


# todo このさきを作り変えること
'''
## make text into vector ##
seq_text_encoded_obj = encoder_generator.encode_text(
    seq_input_text_object=seq_test_input_text_obj)  # type: List[EncodedVectorObject]
## Example application: it takes similarity with cosine similarity ##
from scipy.spatial.distance import cosine
seed_text_id = 0
seed_encoded_vector_obj = seq_text_encoded_obj[seed_text_id]
print('seed-text={}'.format(seed_encoded_vector_obj.args['text']))
seq_similarity_pair = [
    (
        encoded_vector_obj,
        cosine(seed_encoded_vector_obj.vector, encoded_vector_obj.vector)
    )
    for i, encoded_vector_obj in enumerate(seq_text_encoded_obj)
    if not i == seed_text_id]
seq_similarity_pair = sorted(seq_similarity_pair, key=lambda simi_tuple:simi_tuple[1])[:10]
for similarity_tuple_obj in seq_similarity_pair:
    print("cosine distance={} pair-text={}".format(similarity_tuple_obj[1], similarity_tuple_obj[0].args['text']))'''