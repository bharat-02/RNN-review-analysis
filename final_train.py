"""Final SimpleRNN training — best config E. SimpleRNN-only."""
import numpy as np, tensorflow as tf, time, json
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

tf.random.set_seed(42); np.random.seed(42)
max_features=10000; max_len=300; emb=128
(X_train,y_train),(X_test,y_test)=imdb.load_data(num_words=max_features)
Xtr_all=sequence.pad_sequences(X_train,maxlen=max_len)
Xte=sequence.pad_sequences(X_test,maxlen=max_len)
n=len(Xtr_all); nv=int(n*0.2)
Xtr,ytr=Xtr_all[:n-nv],y_train[:n-nv]
Xv,yv=Xtr_all[n-nv:],y_train[n-nv:]
print(f"Train {Xtr.shape} Val {Xv.shape} Test {Xte.shape}", flush=True)

model=Sequential(name="SimpleRNN_optimized")
model.add(Embedding(max_features,emb))
model.add(SimpleRNN(128, activation='tanh', dropout=0.2, recurrent_dropout=0.0))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
opt=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
print(f"Params: {model.count_params()}", flush=True)

cb=[EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=2,min_lr=1e-5,verbose=1),
    ModelCheckpoint('simple_rnn_imdb_optimized.h5',monitor='val_loss',save_best_only=True,verbose=1)]
t0=time.time()
h=model.fit(Xtr,ytr,epochs=15,batch_size=64,validation_data=(Xv,yv),callbacks=cb,verbose=2)
dt=time.time()-t0
print(f"Training time: {dt:.1f}s ({dt/60:.1f} min)", flush=True)
# loss curves
plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.plot(h.history['loss'],label='train'); plt.plot(h.history['val_loss'],label='val')
plt.title('Loss'); plt.xlabel('epoch'); plt.legend(); plt.grid(True)
plt.subplot(1,2,2); plt.plot(h.history['accuracy'],label='train'); plt.plot(h.history['val_accuracy'],label='val')
plt.title('Accuracy'); plt.xlabel('epoch'); plt.legend(); plt.grid(True)
plt.tight_layout(); plt.savefig('loss_curves_optimized.png',dpi=120)
print("saved loss_curves_optimized.png", flush=True)
# reload best checkpoint to verify
from tensorflow.keras.models import load_model
best=load_model('simple_rnn_imdb_optimized.h5')
for name,Xx,yy in [("train",Xtr,ytr),("val",Xv,yv),("test",Xte,y_test)]:
    loss,acc=best.evaluate(Xx,yy,verbose=0,batch_size=256)
    p=(best.predict(Xx,verbose=0,batch_size=256).ravel()>0.5).astype(int)
    print(f"{name}: loss={loss:.4f} acc={acc:.4f} prec={precision_score(yy,p):.4f} rec={recall_score(yy,p):.4f} f1={f1_score(yy,p):.4f}", flush=True)
    print(confusion_matrix(yy,p), flush=True)
# NaN/Inf check
preds=best.predict(Xte[:1000],verbose=0).ravel()
print(f"pred NaN={np.isnan(preds).sum()} Inf={np.isinf(preds).sum()} min={preds.min():.4f} max={preds.max():.4f}", flush=True)
# save history
json.dump({k:[float(x) for x in v] for k,v in h.history.items()}, open('history_optimized.json','w'), indent=1)
print("saved history_optimized.json", flush=True)
