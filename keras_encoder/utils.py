#! -*- coding: utf-8 -*-
# typing
from typing import Union, List, Callable, Dict


class InputTextObject(object):
    """Training, encodingの両方のステップで利用するオブジェクト
    1テキストの情報を保持する
    """
    def __init__(self, text_id:Union[str,int], text:str, seq_token:List[Union[str,None]], **args):
        self.text_id = text_id
        self.text = text
        self.args = args
        self.seq_token = seq_token


def preprocess_text(text_index:int,
                    func_get_token:Callable[[str],List[str]],
                    dict_wikipedia_article_obj:Dict[str,str],
                    sentence_eos:str='。',
                    min_sentence_length:int=10)->List[InputTextObject]:
    """* What you can do
    - It cuts text into sentences.
    - It makes sequence of InputTextObject, which is input object
    """
    seq_input_text_obj = []
    for sentence_index, sentence in enumerate(dict_wikipedia_article_obj['text'].strip().split(sentence_eos)):
        sentence_text = sentence.strip() + sentence_eos
        if len(sentence_text) < min_sentence_length:
            continue

        seq_input_text_obj.append(
            InputTextObject(
                text_id='{}-{}'.format(text_index, sentence_index),
                text=sentence_text,
                seq_token=func_get_token(sentence_text)
            ))
    return seq_input_text_obj