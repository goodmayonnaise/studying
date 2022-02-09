import os
from collections import Counter
from konlpy.tag import Okt
import sentencepiece as spm
from typing import Tuple, List, Dict

okt = Okt()

class Tokenizer:
    def __init__(self, config) -> None:
        self.tokenizer_type = config.tokenizer_type

        if self.tokenizer_type == 'konlpy':
            self.min_freq = config.min_freq
            self.stop_word = config.stop_word
            self.input_file = config.input_file
            self.word2idx, self.idx2word = self.konlpy()
            
        else:
            model_prefix = config.vocab_path + config.tokenizer_type + '_' + config.vocab_size
            if os.path.isfile(model_prefix + '.model'):
                self.sp = spm.SentencePieceProcessor()
                self.sp.load(model_prefix + '.model')

            else:

                self.special_token = ' --pad_id'
                self.cmd = ''
                spm.SentencePieceTrainer.Train(self.cmd)
                self.sp = spm.SentencePieceProcessor()
                self.sp.Load(model_prefix + '.model')



    def konlypy(self):
        # PAD UNK SOS EOS 
        word2idx = {'[PAD]' : 0, '[UNK]':1, '[SOS]':2, '[EOS]':3}
        idx2word = {0:'[PAD]', 1:'[UNK]', 2:'[SOS]', 3:'[EOS}'}

        idx = len(word2idx) - 1
        
        cnt = Counter() # key 개수 
        with open(self.input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                sentence, label = line.split('\t')
                token_lst = okt.morphs(sentence.strip(), stem=True) # 형태소로 분리 
                token_lst = [ token for token in token_lst if token not in self.stop_word ]
                cnt.update(token_lst) # ['나', '느','ㄴ'] --> {'나':1,'느':1,'ㄴ':1}

        for k, v in cnt.items():
            if v >= self.min_freq:
                word2idx[k] = idx
                idx2word[idx] = k
                idx += 1

        assert len(word2idx) == len(idx2word) , 'The length of word2idx and idx2word is different'
        return word2idx, idx2word




        

