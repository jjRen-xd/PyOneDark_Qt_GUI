import sys
import torch
import torch.nn as nn
from torchsummary import summary
import numpy as np
import h5py
from sklearn.model_selection import train_test_split as tts

class conv_net(nn.Module):
	def __init__(self,num_classes):
		super(conv_net,self).__init__()

		self.Block1 = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size = 3, padding = 1),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.Block2 = nn.Sequential(
			nn.Conv2d(32, 32, kernel_size = 3, padding = 1),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2)
			)


		self.Block3 = nn.Sequential(
			nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.Block_1 = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=(5, 2), stride=(1, 2), padding=(2, 1)),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		)
		self.Block_2 = nn.Sequential(
			nn.Conv2d(32, 32, kernel_size=(5, 1), stride=1, padding=(2, 0)),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
		)
		self.Block_3 = nn.Sequential(
			nn.Conv2d(32, 16, kernel_size=(5, 1), stride=1, padding=0),
			nn.BatchNorm2d(16),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
		)


		self.flatten = nn.Flatten()
		self.fc1 = nn.Linear(1248, 512)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(0.8)
		self.fc2 = nn.Linear(512, 256)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(0.8)
		self.fc3 = nn.Linear(256,num_classes)

	def forward(self, data):
		x = data[:,:,:1024].reshape(-1,1,32,32)
		x = self.Block1(x)
		x = self.Block2(x)
		x = self.Block3(x)
	

		y = data[:,:,1024:].reshape(-1,1,128,2)
		y = self.Block_1(y)
		y = self.Block_2(y)
		y = self.Block_3(y)


		x = self.flatten(x)
		y = self.flatten(y)
		x = torch.cat((x,y),dim = 1)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.dropout(x)
		x = self.fc2(x)
		x = self.relu(x)
		x = self.dropout(x)
		x = self.fc3(x)
		return x
class conv_net_harr(nn.Module):
	def __init__(self,num_classes,channel):
		self.channel = channel
		super(conv_net_harr,self).__init__()

		self.Block1 = nn.Sequential(
			nn.Conv2d(self.channel, 32, kernel_size = 3, padding = 1),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.Block2 = nn.Sequential(
			nn.Conv2d(32, 32, kernel_size = 3, padding = 1),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2)
			)


		self.Block3 = nn.Sequential(
			nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.Block_1 = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=(5, 2), stride=(1, 2), padding=(2, 1)),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		)
		self.Block_2 = nn.Sequential(
			nn.Conv2d(32, 32, kernel_size=(5, 1), stride=1, padding=(2, 0)),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
		)
		self.Block_3 = nn.Sequential(
			nn.Conv2d(32, 16, kernel_size=(5, 1), stride=1, padding=0),
			nn.BatchNorm2d(16),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
		)


		self.flatten = nn.Flatten()
		self.fc1 = nn.Linear(1248, 512)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(0.8)
		self.fc2 = nn.Linear(512, 256)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(0.8)
		self.fc3 = nn.Linear(256,num_classes)

	def forward(self, data):
		x = data[:,:,:1024*self.channel].reshape(-1,self.channel,32,32)
		x = self.Block1(x)
		x = self.Block2(x)
		x = self.Block3(x)
	

		y = data[:,:,1024*self.channel:].reshape(-1,1,128,2)
		y = self.Block_1(y)
		y = self.Block_2(y)
		y = self.Block_3(y)


		x = self.flatten(x)
		y = self.flatten(y)
		x = torch.cat((x,y),dim = 1)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.dropout(x)
		x = self.fc2(x)
		x = self.relu(x)
		x = self.dropout(x)
		x = self.fc3(x)
		return x
class conv_net_dqn(nn.Module):
	def __init__(self,num_classes):
		super(conv_net_dqn,self).__init__()

		self.Block1 = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size = 3, padding = 1),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.Block2 = nn.Sequential(
			nn.Conv2d(32, 32, kernel_size = 3, padding = 1),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2)
			)


		self.Block3 = nn.Sequential(
			nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2)
			)


		self.flatten = nn.Flatten()
		self.fc1 = nn.Linear(1028, 512)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(0.8)
		self.fc2 = nn.Linear(512, 256)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(0.8)
		self.fc3 = nn.Linear(256,num_classes)


	def forward(self, data):
		x = data[:,:,:1024].reshape(-1,1,32,32)
		x = self.Block1(x)
		x = self.Block2(x)
		x = self.Block3(x)
	

		y = data[:,:,1024:]


		x = self.flatten(x)
		y = self.flatten(y)
		x = torch.cat((x,y),dim = 1)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.dropout(x)
		x = self.fc2(x)
		x = self.relu(x)
		x = self.dropout(x)
		x = self.fc3(x)

		return x

def dataharr(obs):
	path = r".\data\stars_haar_level2.h5"
	h5file = h5py.File(path, 'r')
	index = []
	for i in obs:
		if i:
			index.append(True)
		else:
			index.append(False)

	datas = h5file['datas'][:]
	labels = h5file['labels'][:]
	stars = h5file['stars'][:][:,index,:,:]
	stars_train = stars.reshape(-1,1,1024*int(np.sum(obs)))
	datas_train = datas.reshape(-1,1,256)
	X_train = np.concatenate((stars_train,datas_train),axis =2)
	# X_train = datas[snrs==snr].reshape(-1,1,128,2)
	Y_train = labels
	x_train, x_test, y_train, y_test = tts(X_train, Y_train, random_state = 1121,train_size = 0.2, stratify = Y_train)
	
	return x_train,y_train,x_test,y_test


if __name__ == '__main__':
	x = np.array([125,124, 141, 249, 41, 141,  62,  83, 156, 249, 249.])
	print(np.sum(x*x)/(1620**2))