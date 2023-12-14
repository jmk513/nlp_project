from fasttext_custom import *

corpus_modu = 'data/modu_spoken_v1.0.txt'

lemmed_corpus_modu = 'data/lem_modu_spoken_v1.0.txt'

corpus_pciko = 'data/pciko_corpus.txt'

lemmed_corpus_pciko = 'data/lem_pciko_corpus.txt'

pre_model = 'model/modu_spoken_100_min5_win10.model'

trained_model = 'model/pciko_10_finetuned.model'


print("lemm_start!")

lemmatization(corpus_modu, lemmed_corpus_modu)
lemmatization(corpus_pciko, lemmed_corpus_pciko)

print("lemm_done!")

train_fasttext(lemmed_corpus_modu, 100, pre_model)

update_pretrained(lemmed_corpus_pciko, pre_model, trained_model)

print("train_done!")

test(trained_model, "공")
test(trained_model, "숟가락")
test(trained_model, "포크")
test(trained_model, "양말")



