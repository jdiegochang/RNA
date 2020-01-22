#-------------------------------Imports-------------------------------#
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Dense, Flatten
#-------------------------------Dataset--------------------------------#
def dataset(categories):
#-------------------MNIST Loading------------------------#
  (x_trn, y_trn), (x_tst, y_tst) = mnist.load_data()
#--------------Float and normalization-------------------#
  x_trn = x_trn.astype('float32')
  x_tst = x_tst.astype('float32')
  x_trn /= 255
  x_tst /= 255
#-----------Transformation to 4 Dimensions---------------#
#(60k, 28, 28, 1) 60K images, 28 height, 28 width, 1 channel
#(n, width, height, depth)
  x_trn = x_trn.reshape(x_trn.shape[0], 28, 28, 1)
  x_tst = x_tst.reshape(x_tst.shape[0], 28, 28, 1)
#------------------One-Hot Encoding----------------------#
  y_trn = to_categorical(y_trn, categories)
  y_tst = to_categorical(y_tst, categories)
  return (x_trn,x_tst,y_trn,y_tst)
#-----------------------Network Architecture-------------------------#
def architecture(shape):
  model = Sequential()
  model.add(Conv2D(filters=18, kernel_size=3, strides=(1, 1), activation='relu', input_shape=shape))
  model.add(MaxPool2D())
  model.add(Conv2D(filters=42, kernel_size=3, strides=(1, 1), activation='relu', input_shape=shape))
  model.add(MaxPool2D())
  model.add(Flatten())
  model.add(Dense(units=120, activation='relu'))
  model.add(Dense(units=84, activation='relu'))
  model.add(Dense(units=10, activation='softmax'))
  return model
###########################-------MAIN--------##################################
if __name__ == "__main__":
  x_train,x_test,y_train,y_test=dataset(10)

  cnn=architecture((28,28,1))
  cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  print(cnn.summary())
  cnn.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3)