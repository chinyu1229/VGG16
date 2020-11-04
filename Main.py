import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton,QMainWindow
from PyQt5.QtGui import QPixmap,QImage

import tensorflow as tf
import numpy as np
import random
from numpy import argmax

from UI5 import Ui_Form



(train_images,train_labels),(test_images,test_labels) = tf.keras.datasets.cifar10.load_data()
label_dict=["plain","car","bird","cat","deer","dog","frog","horse","ship","truck"]


train_images,test_images = train_images / 255.0, test_images / 255.0


class MyMainForm(QMainWindow, Ui_Form):
	def __init__(self,parent = None):
		super(MyMainForm, self).__init__(parent)
		self.setupUi(self)
		self.pushButton1.clicked.connect(self.Q1)
		self.pushButton2.clicked.connect(self.Q2)
		self.pushButton3.clicked.connect(self.Q3)
		self.pushButton4.clicked.connect(self.Q4)
		self.pushButton5.clicked.connect(self.Q5)

	def closeEvent(self, event):
		sys.exit(app.quit())

	def Q1(self):
		ridx = random.sample(range(0,len(train_labels)),10)
		for i in range(10):
			ax=plt.subplot(3,4,i+1)
			ax.imshow(train_images[ridx[i]],cmap=plt.cm.binary)
			title=label_dict[train_labels[ridx[i]][0]]
			ax.set_title(title,fontsize=10)
			ax.axis('off')

		plt.show()

	def Q2(self):
		print('hyper parameters:')
		print('batch size: 32')
		print('learning rate: 0.001')
		print('optimizer: SGD')

	def Q3(self):
		vgg16 = tf.keras.applications.VGG16(include_top=False,weights='imagenet',input_shape=(32,32,3))
		model = tf.keras.Sequential()
		model.add(vgg16)
		model.add(tf.keras.layers.Flatten())
		model.add(tf.keras.layers.Dense(4096,activation='relu'))
		model.add(tf.keras.layers.Dense(4096,activation='relu'))
		model.add(tf.keras.layers.Dense(10,activation='softmax'))


		print(model.summary())
	
	def Q4(self):
		#train
		"""
		model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate = 0.001),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),metrics=['accuracy'])


		history = model.fit(train_images,train_labels,epochs=40,batch_size=32,validation_data=(test_images,test_labels))

		model.save('my_model.h5')

		plt.figure()
		subplot(2,1,1)
		plt.plot(history.history['accuracy']*100,label='Training')
		plt.plot(history.history['val_accuracy']*100,label='Testing')
		plt.title('Accuracy')
		plt.ylabel('%')
		plt.ylim([0,100])
		plt.legend(loc='lower right')

		subplot(2,1,2)
		plt.plot(history.history['loss'])
		plt.xlabel('Epoch')
		plt.ylabel('loss')
		"""
		img = cv2.imread('./acc_loss.png')
		cv2.namedWindow('acc & loss',0)
		cv2.imshow('acc & loss',img)
		cv2.waitKey(0)
		cv2.destroyWindow('acc & loss')

	def Q5(self):
		val = self.spinBox.value()
		model = tf.keras.models.load_model('./my_model.h5')
		predict_img=test_images[val].reshape(1,32,32,3)
		#predict_label=argmax(test_labels[0])
		predict_result=model.predict(predict_img)
		plt.figure()
		plt.xticks([])
		plt.yticks([])
		plt.grid(False)
		plt.imshow(test_images[val], cmap=plt.cm.binary)
		#plt.xlabel(label_dict[test_labels[0][0]],fontsize=16)

		fig=plt.figure()
		ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
		plt.bar(range(10),predict_result[0])
		plt.grid(False)
		plt.xticks(range(10))
		plt.ylim([0, 1])
		plt.yticks([i/10 for i in range(11)])
		ax.set_xticklabels(label_dict, fontsize=10)
		plt.show()



if __name__ == "__main__":
	app = QApplication(sys.argv)
	myWin = MyMainForm()
	myWin.show()
	sys.exit(app.exec_())
