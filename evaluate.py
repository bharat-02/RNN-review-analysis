"""Evaluate SimpleRNN model on IMDB test set. Usage: python evaluate.py [--model FILE]"""
import sys, numpy as np, tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
path = sys.argv[sys.argv.index('--model')+1] if '--model' in sys.argv else 'simple_rnn_imdb_optimized.h5'
model = load_model(path)
maxlen = model.input_shape[1] or 300
print(f"model={path} maxlen={maxlen} params={model.count_params()}")
(X_train,y_train),(X_test,y_test)=imdb.load_data(num_words=10000)
Xte = sequence.pad_sequences(X_test, maxlen=maxlen)
loss,acc = model.evaluate(Xte, y_test, verbose=0, batch_size=256)
p = (model.predict(Xte, verbose=0, batch_size=256).ravel() > 0.5).astype(int)
print(f"test loss={loss:.4f} acc={acc:.4f} prec={precision_score(y_test,p):.4f} rec={recall_score(y_test,p):.4f} f1={f1_score(y_test,p):.4f}")
print(confusion_matrix(y_test,p))
assert np.isnan(p).sum()==0
