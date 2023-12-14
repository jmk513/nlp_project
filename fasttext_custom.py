#-*- coding:utf-8 -*-

from gensim.models.fasttext import FastText, load_facebook_model
from gensim.models import Word2Vec
from gensim.test.utils import datapath

from konlpy.tag import Komoran

import argparse




def train_fasttext(corpus_file, epoch, model_save):

    model = FastText(min_count=1, sg=1)

    # build the vocabulary
    model.build_vocab(corpus_file=corpus_file)

    # train the model
    model.train(
        corpus_file=corpus_file, epochs=epoch,
        total_examples=model.corpus_count, total_words=model.corpus_total_words,
    )

    model.save(model_save)

def test(model, word):
    loaded_model = FastText.load(model)

    wv = loaded_model.wv

    similar = [w[0] for w in wv.most_similar(word)]
    
    print(similar)

def lemmatization_khaiii(open_file, write_file):
    api = KhaiiiApi()

    with open(open_file,'rt', encoding='utf-8') as f :
        with open(write_file,'wt', encoding='utf-8') as wf:

            for line in f.readlines():
                try:
                    line_pos = api.analyze(line)
                        
                except Exception:
                    pass
                
                for w in line_pos:
                    for m in w.morphs:
                        if m.tag in ['NNG', 'NR', 'MAG']:
                            wf.write(m.lex +' ')
                        elif m.tag in ['VV', 'VA']:
                            wf.write(m.lex + '다 ')
                        elif m.tag in ['XR']:
                            wf.write(m.lex+'하다 ')
                        elif m.tag in ['SF','SP', 'MAC']:
                            wf.write('\n')
                        else :
                            continue
    


def lemmatization(open_file, write_file):
    Komo = Komoran()

    with open(open_file,'rt', encoding='utf-8') as f :
        with open(write_file,'wt', encoding='utf-8') as wf:

            for line in f.readlines():
                try:
                    line_pos = Komo.pos(line)
                    
                    for w in line_pos:
                        if w[1] in ['NNG', 'NNP', 'NR', 'MDT', 'MDN', 'MAG']:
                            wf.write(w[0] +' ')
                        elif w[1] in ['VV', 'VA']:
                            wf.write(w[0] + '다 ')
                        elif w[1] in ['XR']:
                            wf.write(w[0]+'하다 ')
                        elif w[1] in ['SF','SP', 'MAC']:
                            wf.write('\n')
                        else :
                            continue
                        
                except Exception:
                    pass          
                             

def update_pretrained(corpus_file, model_pretrained, model_save):

    pretrained_model = FastText.load(model_pretrained)

    pretrained_model.build_vocab(corpus_file=corpus_file, update= True)

    pretrained_model.train(
        corpus_file=corpus_file, epochs=10,
        total_examples=pretrained_model.corpus_count, total_words=pretrained_model.corpus_total_words,
    )

    pretrained_model.save(model_save)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('word', type=str)


    args = parser.parse_args()

    test(args.model, args.word)
