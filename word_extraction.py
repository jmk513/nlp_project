from filter_wv import *
import numpy as np


model_path = "model/pciko_10_finetuned.model"

vocab_path = "data/wv_1052_pciko_10_finetuned.npy"


def get_word_list (target_word_list, word_n):

    target_words = np.array(target_word_list).T
    
    ## get word vector dataset
    vocab_wv = np.load(vocab_path, allow_pickle=True)
    
    ## result word number for each target word
    word_num = word_n*target_words[1].astype(float)
    
    ## word vectors for target words : [[word1, word1.vec], [word2, word2.vec], ...]
    target_words = target_words[0]
    target_wv = get_wv(model_path, target_words)
    
    
    word_list = []
    for target, num in zip(target_wv, word_num):
        
        ## compute similarity w/ word vector dataset
        sim = get_sim(vocab_wv, target[1])
        
        ## extract top-n similarity words
        sim_words = top_n(target[0], sim, int(num))
        
        word_list.append({target[0]:sim_words})
        
    return word_list
        
    
    
    
if __name__ == "__main__" :
    scenario = [
        ['강아지', 1],
        ['신발', 1],
        ['자전거', 1],
        ['버스', 1],
        ['가방', 1],
        ['나비', 1],
    ]
    
    word_n = 5
    
    word_list = get_word_list(scenario, word_n)
    
    for w in word_list:
        print(w)


    
    


