from word_extraction import *
from api_call import *

target_words = [
    ['강아지', 1],
    ['신발', 1],
    ['자전거', 1],
    ['버스', 1],
    ['가방', 1],
    ['나비', 1],
]

word_n = 5

related_word_dict = get_word_list(target_words, word_n)

for d in related_word_dict:
    generate_sentences_from_dict(d)