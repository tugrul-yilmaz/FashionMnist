####################################################
#Fashion MNIST
####################################################
#1)Kütüphanlerin Yüklenmesi ve Veriye Genel Bakış
#2)Veri Ön İşleme
#3)Model Oluşturulması
#4)Train
#5)Test


##########################################################################################################################
#1)Kütüphanlerin Yüklenmesi ve Veriye Genel Bakış

import matplotlib.pyplot as plt
import keras


data.head()
data.shape


train=data.iloc[:,1:]
train.shape

label=data["label"]
label.shape

plt.figure(figsize=(16,5))
for i in range(1,12):
    plt.subplot(3,4,i)
    plt.imshow(train.iloc[i].values.reshape(28,28),cmap="gray")

##########################################################################################################################
#2) Veri Ön İşleme

num_classes=label.unique()
print(num_classes)

train_label=keras.utils.to_categorical(label,len(num_classes))
print(train_label.shape)
print(type(train_label))

train=train.values
train=train.reshape(-1,28,28,1)
print(train.shape)
print(type(train))

train=train.astype("float32")
train=train/255
print(train.shape)

from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val=train_test_split(train,train_label,test_size=0.1,random_state=1)
print(x_train.shape)
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)

from keras.preprocessing.image import ImageDataGenerator
ll=datagen.fit(x_train)


##########################################################################################################################
#3)Model Oluşturulması

from keras.models import Sequential,load_model
from keras.layers import Conv2D,Dense,Flatten
from keras.layers import MaxPool2D,BatchNormalization,Dropout

model=Sequential()

# 1.katman
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))
model.add(BatchNormalization())

# 2.katman
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.1))

# 3.katman
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(BatchNormalization())

# 4.katman
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

# Tam bağlantı katmanı
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))

# Çıkış katmanı
model.add(Dense(10, activation = "softmax"))


model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

model.summary()


##########################################################################################################################
#4)Train

hist=model.fit_generator(datagen.flow(x_train,y_train,batch_size=64),
                           epochs=30,
                           shuffle=True,
                           validation_data=(x_val,y_val))


plt.figure(figsize=(15,3))
plt.subplot(1,2,1)
plt.plot(hist.history["loss"],"r",label="Training Loss",alpha=0.5)
plt.plot(hist.history["val_loss"],"b",label="Validation Loss",alpha=0.5)
plt.xlabel("Train",fontsize=13)
plt.ylabel("Loss",fontsize=13)
plt.legend(loc="upper right")

plt.figure(figsize=(15,3))
plt.subplot(1,2,2)
plt.plot(hist.history["accuracy"],"g",label="Training Accuracy",alpha=0.8)
plt.plot(hist.history["val_accuracy"],"m",label="Validation Accuracy",alpha=0.8)
plt.xlabel("Train",fontsize=13)
plt.ylabel("Accuracy",fontsize=13)
plt.legend(loc="lower right")

##########################################################################################################################
#5)Test

test=pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_test.csv")
test.head()

x_test=test.iloc[:,1:]
x_test.shape
y_test=test.iloc[:,0]
y_test.shape

x_test=x_test.values.reshape(-1,28,28,1)
x_test=x_test/255

y_test=keras.utils.to_categorical(y_test,len(num_classes))
y_test.shape

labels=test.iloc[:,0]
labels.shape

images=test.iloc[:,1:]
images.shape

images=images.values
labels=labels.values

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

lp=int(input("Tahmin etmek istediğiniz resimin numarasını giriniz"))
predicted=model.predict(x_test[lp].reshape(-1,28,28,1))
ls=np.argmax(predicted)

plt.figure(figsize=(12,3))
plt.imshow(images[lp].reshape(28,28),cmap="gray")
plt.xlabel("Tahmininiz"+" "+str(class_names[ls]))