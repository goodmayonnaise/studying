from tqdm.notebook import tqdm
from config import Config
import os
import sentencepiece as spm
from konlpy.tag import Kkma
from collections import Counter
import numpy as np
from nltk.tokenize import word_tokenize
import torch

class Tokenizer:
    def __init__(self, config):
        self.config = config
        if self.config.tokenizer_type in ['bpe', 'char', 'bigram']:
            self.sp = spm.SentencePieceProcessor()
            self.prefix = self.config.vocab_path + self.config.tokenizer_type + '_' + str(self.config.vocab_size)
            if os.path.isfile(self.prefix + '.model'):
                self.sp.Load(self.prefix + '.model')
                
            else:
                self.train()

        elif self.config.tokenizer_type == 'konlpy':
            self.kkma = Kkma()
            self.word2idx, self.idx2word = self.konlpy()

        else:
            self.word2idx, self.idx2word = self.nltk_word_tokenize()

    def konlpy(self):
        PAD = '[PAD]'
        UNK = '[UNK]'
        BOS = '[BOS]'
        EOS = '[EOS]'

        word2idx = {PAD:0, UNK:1, BOS:2, EOS:3}
        idx2word = {0:PAD, 1:UNK, 2:BOS, 3:EOS}
        
        idx = len(word2idx)
        max_len = 0
        count_dict = Counter()
        for line in open(self.config.vocab_infile_path):

            line = line.strip()
            line = self.kkma.morphs(line)
            max_len = max(max_len, len(line))
            line = [ word for word in line if word not in self.config.stop_word ]
            count_dict.update(line)

        for k,v in count_dict.items():
            if v >= self.config.min_freq:
                word2idx[k] = idx
                idx2word[idx] = k
                idx += 1

        print(f"[Konlpy] Max length : {max_len}")
        return word2idx, idx2word

    def nltk_word_tokenize(self):
        PAD = '[PAD]'
        UNK = '[UNK]'
        BOS = '[BOS]'
        EOS = '[EOS]'

        word2idx = {PAD:0, UNK:1, BOS:2, EOS:3}
        idx2word = {0:PAD, 1:UNK, 2:BOS, 3:EOS}
        
        idx = len(word2idx)
        max_len = 0
        count_dict = Counter()
        for line in open(self.config.vocab_infile_path + 'vocab_data.txt'):

            line = line.strip()
            line = word_tokenize(line)
            max_len = max(max_len, len(line))
            line = [ word for word in line if word not in self.config.stop_word ]
            count_dict.update(line)

        for k,v in count_dict.items():
            if v >= self.config.min_freq:
                if k not in word2idx:
                    word2idx[k] = idx
                    idx2word[idx] = k
                    idx += 1

        print(f"[NLTK Tokenize] Max length : {max_len}")
        return word2idx, idx2word

    def load_pretrained_vectors(self):
        fin = open(self.config.pretrained_file, 'r', encoding='utf-8', errors='ignore', newline='\n')
        n, d = map(int, fin.readline().split())

        # random embeddings
        embeddings = np.random.uniform(-0.25, 0.25, (len(self.word2idx), d))
        embeddings[self.word2idx['[PAD]']] = np.zeros=((d,))

        # Load vectors
        count = 0
        for line in tqdm(fin):
            tokens = line.rstrip().split()
            word = tokens[0]
            if word in self.word2idx:
                count += 1
                embeddings[self.word2idx[word]] = np.array(tokens[1:], dtype=np.float32)
        
        print(f"There are {count} / {len(self.word2idx)} pretrained vectors found.")
        return torch.tensor(embeddings)


    def train(self):
        spm.SentencePieceTrainer.train(
            f"--input={self.config.vocab_infile_path + 'vocab_data.txt'} --model_prefix={self.prefix} --vocab_size={self.config.vocab_size+4}" +
            f" --model_type={self.config.tokenizer_type}" +
            ' --character_coverage=1.0 --shuffle_input_sentence=true' +
            ' --max_sentence_length=999999' +
            ' --pad_id=0 --pad_piece=[PAD]' +
            ' --unk_id=1 --unk_piece=[UNK]' +
            ' --bos_id=2 --bos_piece=[BOS]' +
            ' --eos_id=3 --eos_piece=[EOS]' +
            f' --user_defined_symbols={self.config.user_symbols}'
        )
        self.sp.Load(self.prefix + '.model')

    def encodeAsids(self, text):
        if self.config.tokenizer_type == 'konlpy':
            line = self.kkma.morphs(text)
            return np.array([self.word2idx[word] if word in self.word2idx else self.word2idx['[UNK]'] for word in line])

        elif self.config.tokenizer_type == "word_tokenize":
            line = word_tokenize(text)
            return np.array([self.word2idx[word] if word in self.word2idx else self.word2idx['[UNK]'] for word in line])
        return self.sp.EncodeAsIds(text)

    def encodeAspieces(self, text):
        if self.config.tokenizer_type == 'konlpy':
            return self.kkma.morphs(text)
        elif self.config.tokenizer_type == "word_tokenize":
            return word_tokenize(text)
        return self.sp.EncodeAsPieces

    def decode(self, ids):
        if self.config.tokenizer_type == 'konlpy' or self.config.tokenizer_type == "word_tokenize":
            line = [self.idx2word[id] for id in ids]
            return ' '.join(line)
        return self.sp.Decode(ids)

    def pad_id(self):
        if self.config.tokenizer_type == 'konlpy' or self.config.tokenizer_type == "word_tokenize":
            return self.word2idx['[PAD]']
        return self.sp.pad_id()

    def get_size(self):
        if self.config.tokenizer_type == 'konlpy' or self.config.tokenizer_type == "word_tokenize":
            return len(self.word2idx)
        return self.sp.GetPieceSize()

if __name__ == "__main__":
    config = Config("/Users/khs/Desktop/김현수/studying/projects/SentimentClassification/hyunsoo/config.json")
    tok = Tokenizer(config)
    # print(tok.sp.piece_size())
    # print(tok.sp.EncodeAsPieces('너의 손 꼭잡고 그냥 이 길을 걸었으면 내게 너뿐인걸 꼭 니가 알아줬으면 좋을텐데'))
    print(tok.encodeAsids('캐스팅과'))
    print(tok.decode(tok.encodeAsids('바스코갯웃기네 캐스팅과 질퍽하지 않은')))

    embeddings=tok.load_pretrained_vectors()
    print(embeddings.size())
    print(tok.pad_id())    