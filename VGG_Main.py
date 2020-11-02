from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import random


(train_images,train_labels),(test_images,test_labels) = tf.keras.datasets.cifar10.load_data()
label_dict={0:"airplain",1:"automobile",2:"bird",3:"cat",4:"deer",5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}



#Q1

ridx = random.sample(range(0,len(train_labels)),10)
for i in range(10):
	ax=plt.subplot(3,4,i+1)
	ax.imshow(train_images[ridx[i]],cmap=plt.cm.binary)
	title=label_dict[train_labels[ridx[i]][0]]
	ax.set_title(title,fontsize=10)
	ax.axis('off')

plt.show()

vgg16 = tf.keras.applications.VGG16(include_top=False,weights='imagenet',input_shape=(32,32,3))
model = tf.keras.Sequential()
model.add(vgg16)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64,activation='relu'))
model.add(tf.keras.layers.Dense(32,activation='relu'))
model.add(tf.keras.layers.Dense(10))


print(model.summary())

#train

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate = 0.0001),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),metrics=['accuracy'])

history = model.fit(train_images,train_labels,epochs=20,batch_size=32,validation_data=(test_images,test_labels))

model.save('my_model.h5')

plt.figure()
plt.subplot(2,1,1)
history.history['accuracy'] = [x*100 for x in history.history['accuracy']
history.history['val_accuracy'] = [x*100 for x in history.history['val_accuracy']
plt.plot(history.history['accuracy'],label='Training')
plt.plot(history.history['val_accuracy'],label='Testing')
plt.title('Accuracy')
plt.ylabel('%')
plt.ylim([0,100])
plt.legend(loc='lower right')

plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('loss')


