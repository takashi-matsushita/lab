import random

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('bmh') 
import matplotlib
matplotlib.rcParams['font.family'] = 'IPAPGothic'

import seaborn as sns

from sklearn.model_selection import train_test_split
import umap

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger


### set seeds for reproducibility
def reset_seeds():
  random.seed(1111)
  np.random.seed(3333)
  import tensorflow as tf; tf.set_random_seed(5555)
reset_seeds()


### read data set
train_images = np.load('kuzushiji/k49-train-imgs.npz')['arr_0']
test_images = np.load('kuzushiji/k49-test-imgs.npz')['arr_0']
train_labels = np.load('kuzushiji/k49-train-labels.npz')['arr_0']
test_labels = np.load('kuzushiji/k49-test-labels.npz')['arr_0']

### read character code to character mapping
df = pd.read_csv('kuzushiji/k49_classmap.csv')

### create code/character map
k49 = df.set_index('index').to_dict()['char']
n_class = len(k49)

### visualise some characters
for ii in range(5):
  idx = random.randint(0, len(train_images))
  label = train_labels[idx]
  char = k49[label]
  print('displaying label={} char={}'.format(label, char))
  plt.imshow(train_images[idx], cmap="gray")
  plt.title(char)
  plt.colorbar()
  plt.show()
  input('press return key to continue...')
  plt.close()


### number of training sample per character
labels = df['char'] 
ax = plt.subplots(1,1, figsize=(8,6)) 
g = sns.countplot(train_labels) 
g.set_title("Number of training samples per character") 
g.set_xticklabels(labels) 
plt.show() 

### percentage of each character to total training characters
ratios = np.bincount(train_labels)/len(train_labels)*100
for ii in range(len(ratios)):
  print('{} {:.2f}%'.format(k49[ii], ratios[ii]))

### normalise inputs
train_images = train_images / 255.0
test_images = test_images / 255.0


### 2D embedding
train_sample = train_images.reshape(len(train_images), -1)

def plot_embeddings(title, n_class):
  fig = plt.figure(figsize=(10, 8))
  ax = fig.add_subplot(111)

  sns.set_palette(sns.xkcd_palette(sns.xkcd_rgb), n_colors=n_class)
  marker = [
    'o', # circle
    'x', # X
    '^', # triangle up
    '+', # plus
    '8', # octagon
    's', # square,
    'p', # filled plus
    'D', # diamond
    'H', # Hexagon
    ]
  markers = [marker[ii%len(marker)] for ii in range(n_class)]

  for ii, label in zip(df.index.values, df.char.values):
      sns.scatterplot(embeddings[train_labels == ii, 0], 
                      embeddings[train_labels == ii, 1], 
                      label=label, marker=markers[ii], s=18)
  plt.title(title, fontsize=16)
  plt.legend(ncol=3, loc='upper right', bbox_to_anchor=(1.,1))
  plt.xlim(-15,25)
  plt.ylim(-20,20)
  plt.show()


### UMAP
reducer = umap.UMAP(n_components=2, min_dist=0.8, metric='correlation', init='random', random_state=42, verbose=1)
embeddings = reducer.fit_transform(train_sample)
plot_embeddings("Embeddings with UMAP for Kuzushiji", n_class)


### prepare input set
X = np.expand_dims(train_images, axis=-1)
x_test = np.expand_dims(test_images, axis=-1)
y = keras.utils.to_categorical(train_labels)
y_test = keras.utils.to_categorical(test_labels)

x_train, x_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=1013)
# check class distribution of split sample
sns.set_palette("tab10") 
sns.countplot(np.argmax(y_val, axis=1))


### build model
reset_seeds()
img_rows, img_cols = train_images.shape[1:]
batch_size = 128
epochs = 20

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3),
                 activation='relu',
                 input_shape=(img_rows,img_cols,1)))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_class, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.summary()


### define callbacks
filepath = "cnn-k49-best.h5"
callbacks_list = [
  ModelCheckpoint(filepath, monitor='val_acc', mode='max', save_best_only=True, verbose=1),
  EarlyStopping(monitor='val_acc', mode='max', patience=5),
  CSVLogger(filename='cnn-k49-log.csv'),
  ReduceLROnPlateau(patience=3, verbose=1),
  ]

### fit
hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                 verbose=1, callbacks=callbacks_list,
                 validation_data=(x_test, y_test))
model.save('cnn-k49-last.h5')


### plot loss/accuracy vs epoch
fig, ax1 = plt.subplots()
ax1.plot(hist.history['loss'], 'b-', label='loss') 
ax1.plot(hist.history['val_loss'], 'b--', label='val_loss') 
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.legend(loc='center right', bbox_to_anchor=(1.,0.60))

ax2 = ax1.twinx()
ax2.plot(hist.history['acc'], 'r-', label='acc') 
ax2.plot(hist.history['val_acc'], 'r--', label='val_acc') 
ax2.set_ylabel('accuracy', color='r')
ax2.tick_params(axis='y', labelcolor='r')
ax2.legend(loc='center right', bbox_to_anchor=(1.,0.40))

fig.tight_layout()
plt.show()

###


predict_classes = model.predict_classes(x_test)

# check some of failed predictions
for idx in range(min(100,len(y_test))):
  index = argmax(y_test[idx])
  if index == predict_classes[idx]: continue
  plt.imshow(x_test[idx][:,:,0], cmap="gray")
  true = k49[argmax(y_test[idx])]
  print(index == predict_classes[idx], 'true: ', true, '  guess:', k49[predict_classes[idx]])
  input('press return key to continue...')
  plt.close()


# classification report
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

y_true = np.argmax(y_test,axis=1)
correct = np.nonzero(predict_classes==y_true)[0]
incorrect = np.nonzero(predict_classes!=y_true)[0]
print("Correct predicted classes:",correct.shape[0])
print("Incorrect predicted classes:",incorrect.shape[0])

target_names = ["Class {} ({}):".format(i, df[df['index']==i]['char'].item()) for i in range(n_class)]
print(accuracy_score(y_true, predict_classes))
print(classification_report(y_true, predict_classes, target_names=target_names))
cm = confusion_matrix(y_true, predict_classes)
print(cm)

# plot confusion matrix
cm_normalised = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
df_cm = pd.DataFrame(cm_normalised, df.char.tolist(), df.char.tolist())

fontsize=8
hm = sns.heatmap(df_cm, cmap="jet")
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), fontsize=fontsize)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), fontsize=fontsize)
plt.ylabel('True')
plt.xlabel('Predicted')

# another way of checking confusion matrix
import pycm
cm = pycm.ConfusionMatrix(y_true, predict_classes)
cm.save_csv('cm')
df_cm = pd.read_csv('cm.csv')
df_cm = df_cm.set_index('Class')
df_cm = df_cm.transpose()
print(df_cm[['PPV', 'TPR', 'F1']])  # precision, recall, f1

# check classes with f1 value < threshold
f1 = df_cm[['F1', 'PPV', 'TPR']].to_dict()     
threshold = 0.80
print('c  f1   prec. recall')
poor = []
for key, data in f1.items():
  if key != 'F1': continue
  for idx, v in f1[key].items():
    v = float(v)
    if v < threshold:
      poor.append(int(idx))
      print('{} {:.2f} {:.2f} {:.2f}'.format(k49[int(idx)], v, float(f1['PPV'][idx]), float(f1['TPR'][idx])))


# visual check of classes with poor performance
f, axes = plt.subplots(5, 5, sharex=True, sharey=True)
for row in range(5):
  p = axes[row]
  array = np.where(test_labels==poor[row])[0]
  idxs = np.random.choice(array, 5)
  for col in range(5):
    p[col].imshow(test_images[idxs[col]], cmap='gray')
    p[col].axis('off')
    p[col].set_title('{}'.format(k49[poor[row]]))

plt.tight_layout()
plt.show()

# eof
