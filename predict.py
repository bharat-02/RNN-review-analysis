"""Predict sentiment for one review. Usage: python predict.py "your review here" [--model FILE]"""
import sys, re, numpy as np, tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
args=[a for a in sys.argv[1:] if not a.startswith('--')]
text=args[0] if args else "This movie was fantastic! The acting was great and the plot was thrilling."
mp=sys.argv[sys.argv.index('--model')+1] if '--model' in sys.argv else 'simple_rnn_imdb_optimized.h5'
model=load_model(mp); maxlen=model.input_shape[1] or 300
word_index=imdb.get_word_index()
words=re.sub(r"[^a-z0-9 ]"," ",text.lower()).split()
# Rare words outside the top-10000 vocabulary map to OOV (2), mirroring
# imdb.load_data(num_words=10000); larger ids would crash the embedding lookup.
enc=[word_index[w]+3 if (w in word_index and word_index[w]+3 < 10000) else 2 for w in words]
pad=sequence.pad_sequences([enc],maxlen=maxlen)
score=float(model.predict(pad,verbose=0)[0][0])
sent="Positive" if score>0.5 else "Negative"
print(f"Review: {text}\nSentiment: {sent}\nScore: {score:.4f}\nModel: {mp} (maxlen={maxlen})")
