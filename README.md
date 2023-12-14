# nlp_project

## dependencies
```
pip install -r requirements.txt
```

## How to run simulation (process pipeline)?
```
python simulation.py
```
Need following data & models to run simulation
```
data/word_dict_1052.npy
data/wv_1052_pciko_10_finetuned.npy

model/modu_spoken_100_min5_win10.model
model/pciko_10_finetuned.model
model/pciko_10_finetuned.model.wv.vectors_ngrams.npy
```

## modules
### Extract similar words
- fasttext_custom.py : preprocessing functions
- modu_corpus.py : training code for fasttext model
- filter_wv.py : calculates cosine similarities and gets similar words from word dict
- word_extraction.py : extracts top-k similar words
### Generate sentences
- api_call.py : call GPT4 API (needs API key in .env file)
