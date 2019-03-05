import random

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('bmh') 
import matplotlib
matplotlib.rcParams['font.family'] = 'IPAPGothic'

import seaborn as sns

from sklearn.model_selection import train_test_split

import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.models import load_model
from keras.preprocessing import image


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


### normalise inputs
train_images = train_images / 255.0
test_images = test_images / 255.0


### data augmentation, trying to have flat class distribution
classes, counts = np.unique(train_labels, return_counts=True)
idxs = np.where(counts < max(counts))[0]
idx_augment = []
np.random.seed(1013)
for idx in idxs:
  c_idxs = np.where(train_labels==idx)[0]
  nn = max(counts) - counts[idx]
  c_idxs = np.random.choice(c_idxs, nn)
  idx_augment.extend(c_idxs)

n_augment = len(idx_augment)

arguments = {
  'featurewise_center': False,
  'featurewise_std_normalization': False,
  'rotation_range': 20,
  'width_shift_range': 0.1,
  'height_shift_range': 0.1,
  'horizontal_flip': False,
  'vertical_flip': False,
  }
im_gen = image.ImageDataGenerator(**arguments)


### prepare input set
X = np.expand_dims(train_images, axis=-1)
x_test = np.expand_dims(test_images, axis=-1)
y = keras.utils.to_categorical(train_labels)
y_test = keras.utils.to_categorical(test_labels)

x_augmented = X[idx_augment].copy()
y_augmented = y[idx_augment].copy()
x_augmented = im_gen.flow(x_augmented,
                          batch_size=n_augment, shuffle=False).next()

# show some augmented sample
for ii in range(10):
  idx = np.random.randint(n_augment)
  plt.imshow(image.array_to_img(x_augmented[idx]))
  plt.show()
  print(k49[np.argmax(y_augmented[idx])])
  input('press return key to continue...')
  plt.close()

# prepare full set
X = np.concatenate((X, x_augmented))
y = np.concatenate((y, y_augmented))

x_train, x_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=1013)
# check class distribution of split sample
sns.set_palette("tab10") 
sns.countplot(np.argmax(y_val, axis=1))


### load the pre-trained model
model = load_model('cnn-k49-last.h5')

# check learning rate
print(K.get_value(model.optimizer.lr))
# reset learning rate
K.set_value(model.optimizer.lr, 1.0)

"""
### transfer learning
for layer in model.layers:
  if layer.name.find('conv') != -1:
    layer.trainable = False
  if layer.name.find('pooling') != -1:
    layer.trainable = False

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
"""
model.summary()


### define callbacks
filepath = "cnn-k49-augment-best.h5"
callbacks_list = [
  ModelCheckpoint(filepath, monitor='val_acc', mode='max', save_best_only=True, verbose=1),
  EarlyStopping(monitor='val_acc', mode='max', patience=5),
  CSVLogger(filename='cnn-k49-augment-log.csv'),
  ReduceLROnPlateau(patience=3, verbose=1, factor=0.1),
  ]


### fit the model
reset_seeds()
batch_size = 128
epochs = 20
hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                 verbose=1, callbacks=callbacks_list,
                 validation_data=(x_test, y_test))
model.save('cnn-k49-augment-last.h5')


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

df = pd.read_csv('kuzushiji/k49_classmap.csv')
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

# check the five worst classifiers by f1 value
nrow = 5
poor = df_cm.sort_values(by=['F1'], ascending=False)[['F1', 'PPV', 'TPR']][-nrow:]

# get failed predictions for poor performers
to_check = list(map(int, poor.index.tolist()))
indices = {}
for idx in range(len(y_test)):
  cat = argmax(y_test[idx])
  if cat not in to_check: continue
  if cat == predict_classes[idx]: continue

  if cat not in indices: indices[cat] = []
  indices[cat].append((idx, predict_classes[idx]))

# show failed images of poor performers
ncol = 5
row = 0
f, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True)
for cat, _ in poor.iterrows():
  p = axes[row]
  row += 1
  cat = int(cat)
  col = 0
  for idx, guess in indices[cat]:
    p[col].imshow(test_images[idx], cmap='gray')
    p[col].axis('off')
    p[col].set_title('{}: {}'.format(k49[cat], k49[guess]))
    col += 1
    if col == ncol: break

plt.tight_layout()
plt.show()

# eof
