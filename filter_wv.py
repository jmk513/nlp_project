from gensim.models.fasttext import FastText
import numpy as np

def similarity(wv1, wv2):
    n1 = np.linalg.norm(wv1)
    n2 = np.linalg.norm(wv2)
    return np.dot(wv1, wv2)/(n1 * n2)


def get_wv(model_path, vocab_list):
    
    model = FastText.load(model_path)

    wv = model.wv

    
    vocab_wv = [[v, wv[v]] for v in vocab_list]
    
    return vocab_wv

def get_compound_wv(model_path, positive, negative):
    
    model = FastText.load(model_path)
    
    wv = model.wv
    
    positive_wv = [ (wv[word], 1) for word in positive ]
    negative_wv = [ (wv[word], -1) for word in negative ]
    
    
    mean = []
    for word, weight in positive_wv+negative_wv:
        mean.append(word*weight)
        
    compound_wv = np.mean(mean,axis=0)
    
#     print(wv['원숭이'].shape)
    print(compound_wv.shape)
    
    return compound_wv

def get_sim(wv, word):

    sim = [ [v[0],similarity(word, v[1])] for v in wv]
    
    return sim

def top_n(target, sim, n):
    # print(sim[0])
    
    vocab_sorted = sorted(sim, key = lambda s : s[1], reverse = True)
    
    del_index = []
    for i, word in enumerate(vocab_sorted):
        if word[0] in target: del_index.append(i)
            
    vocab_sorted = np.delete(vocab_sorted, del_index, 0)        
            
    
    n_vocab = [v[0] for v in vocab_sorted[:n]]
    
    return n_vocab
    


if __name__ == "__main__" :
    
    model_path = "model/pciko_10_finetuned.model"
    
    vocab_path = "data/word_dict_1052.npy"
    

    vocab = np.load(vocab_path, allow_pickle=True)

    print(vocab)
    
    vocab_wv = np.array(get_wv(model_path, vocab), dtype=object)    
	 
    np.save("data/wv_1052_pciko_10_finetuned.npy", vocab_wv)

    
    
    
    
