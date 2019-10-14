
# In[1]:

from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.layers import Flatten, Dense, Dropout
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
import numpy as np
import os
from random import shuffle
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import pandas as pd
import keras
import PIL


# In[2]:


classmap = pd.read_csv('classmap.csv', header=None, index_col=0)

num_classes = 13
img_size = 224


train_dir = 'dataset/train/'
total_train_cnt = 0
X = []
y = []

dirs = os.listdir(train_dir)
for i_dir in range(len(dirs)):
    dir_name = dirs[i_dir]
    files = os.listdir(train_dir + dir_name)
    cnt_train = len(files)
    print(dir_name, ':', cnt_train)

    for i_file in range(cnt_train):
        if files[i_file].endswith(".jpg"):
            y.append(classmap.loc[dir_name][1])
            img = image.load_img(train_dir + dir_name + '/' + files[i_file],
                                 target_size=(img_size, img_size))
            img = image.img_to_array(img)
            X.append(img)
            total_train_cnt += 1
y = np.asarray(y)
X = np.asarray(X)
print(X.shape)
print(total_train_cnt)


# In[6]:

y = keras.utils.to_categorical(y, num_classes)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=1)
print(X_train.shape)
print(X_val.shape)


# In[11]:


NUM_EPOCHS = 30
BATCH_SIZE = 32


# In[12]:


# width_shift_range & height_shift_range 分別是水平位置評議和上下位置平移
# rotation range 的作用是用戶指定旋轉角度範圍，其參數只需指定一個整數即可
# zoom_range：浮點數或形如[lower,upper]的列表，隨機縮放的幅度
# shear_range就是錯切變換，效果就是讓所有點的x坐標(或者y坐標)保持不變，而對應的y坐標(或者x坐標)則按比例發生平移
# zoom_range參數可以讓圖片在長或寬的方向進行放大，可以理解為某方向的resize
# channel_shift_range可以理解成改變圖片的顏色，通過對顏色通道的數值偏移，改變圖片的整體的顏色
# horizontal_flip的作用是隨機對圖片執行水平翻轉操作

train_datagen = ImageDataGenerator(rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   channel_shift_range=10,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
train_batches = train_datagen.flow(X, y,
                                   shuffle=True,
                                   batch_size=BATCH_SIZE)

valid_datagen = ImageDataGenerator()
valid_batches = valid_datagen.flow(X_val, y_val,
                                   shuffle=False,
                                   batch_size=BATCH_SIZE)


# In[13]:


# 模型輸出儲存的檔案
WEIGHTS_FINAL = 'Final_resnet50_binary_60_all.h5'

FREEZE_LAYERS = 2

net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
               input_shape=(img_size, img_size, 3))
x = net.output
x = Flatten()(x)

# 增加 DropOut layer
x = Dropout(0.5)(x)

# 增加 Dense layer，以 softmax 產生個類別的機率值
output_layer = Dense(num_classes, activation='sigmoid', name='sigmoid')(x)

# 設定凍結與要進行訓練的網路層
net_final = Model(inputs=net.input, outputs=output_layer)
for layer in net_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in net_final.layers[FREEZE_LAYERS:]:
    layer.trainable = True

# 使用 Adam optimizer，以較低的 learning rate 進行 fine-tuning
net_final.compile(optimizer=Adam(lr=1e-5),
                  loss='binary_crossentropy', metrics=['accuracy'])

# 輸出整個網路結構
# print(net_final.summary())

# earlystop
earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
# reduce lr
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=1e-7)

# 訓練模型
model_history = net_final.fit_generator(train_batches,
                                        steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
                                        validation_data=valid_batches,
                                        validation_steps=X_val.shape[0] // BATCH_SIZE,
                                        epochs=NUM_EPOCHS,
                                        callbacks=[earlystop, reduce_lr])

# 儲存訓練好的模型
net_final.save(WEIGHTS_FINAL)


# In[14]:


training_loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

plt.plot(training_loss, label="training_loss")
plt.plot(val_loss, label="validation_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Learning Curve")
plt.legend(loc='best')
plt.show()


# In[15]:


net = load_model(WEIGHTS_FINAL)


# In[33]:


sample_csv = pd.read_csv('sameple_submission.csv')
print(sample_csv.shape)
sample_csv.head()


# In[30]:


sub_id = []
sub_class = []
for i in range(sample_csv.shape[0]):
    ID = sample_csv.loc[i].id
    img = image.load_img('dataset/test/'+ID+'.jpg', target_size=(224, 224))
    if img is None:
        continue
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    pred = net.predict(x)[0]
    pred_y = np.argmax(pred)
#     print(ID, pred_y)
    sub_id.append(ID)
    sub_class.append(classmap.index[pred_y])
#     for i in top_inds:
#         print('    {:.3f}  {}'.format(pred[i], classmap.index[i]))

sub = pd.DataFrame()
sub['id'] = sub_id
sub['label'] = sub_class
sub.to_csv(WEIGHTS_FINAL+'binary.csv', index=False)
