# nlp_project

## dependencies
```
pip install -r requirements.txt
```

## How to run simulation (process pipeline)?
```
python simulation.py
```

## modules
### Extract similar words
- fasttext_custom.py : preprocessing functions
- modu_corpus.py : training code for fasttext model
- filter_wv.py : calculates cosine similarities and gets similar words from word dict
- word_extraction.py : extracts top-k similar words
### Generate sentences
- api_call.py : call GPT4 API (needs API key in .env file)
