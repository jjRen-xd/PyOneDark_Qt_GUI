import numpy as np
import h5py
import random
import matplotlib.pyplot as plt
from app.checkpoints.reinforce import conv_net
import torch
import cv2
from sklearn.model_selection import train_test_split as tts
import os
import sys
signdic = {"GFSK":0, "WBFM":1, "AM-SSB":2, "AM-DSB":3 ,"QPSK":4, "QAM16":5, "CPFSK":6, "BPSK":7, "PAM4":8, "QAM64":9, "8PSK":10}
labeldic = {0:"GFSK", 1:"WBFM", 2:"AM-SSB", 3:"AM-DSB" ,4:"QPSK", 5:"QAM16", 6:"CPFSK", 7:"BPSK", 8:"PAM4", 9:"QAM64", 10:"8PSK"}

def dataharr(root_path, dataset_path, name, obs):
	h5file = h5py.File(dataset_path, 'r')
	index = []
	for i in obs:
		if i:
			index.append(True)
		else:
			index.append(False)
	labels = h5file['labels'][:]
	labels = np.argmax(labels, -1)

	if name not in signdic:
		datas = h5file['datas'][:]
		stars_origin = h5file['stars'][:]
	else:
		datas = h5file['datas'][:][labels == signdic[name]]
		stars_origin = h5file['stars'][:][labels == signdic[name]]
		labels = labels[labels == signdic[name]]
	data_index = random.randint(0,len(datas))

	sign = datas[data_index]
	stars_all = stars_origin[data_index]
	label = labels[data_index]
	stars_chosse = stars_all[index,:,:]

	for i in range(7):
		stars_all[i] = (stars_all[i] - np.min(stars_all[i]))/(np.max(stars_all[i])-np.min(stars_all[i]))
	a, b = np.concatenate((stars_all[0],stars_all[1]),axis = 0),np.concatenate((stars_all[2],stars_all[3]),axis = 0)
	c = np.concatenate((a,b),axis = 1)
	d = np.concatenate((c,cv2.resize(stars_all[4],(64,64))), axis = 0)
	e = np.concatenate((cv2.resize(stars_all[5],(64,64)), cv2.resize(stars_all[6],(64,64))), axis = 0)
	star = np.concatenate((d, e),axis = 1)

	plt.figure()
	plt.axis('off')
	plt.plot(sign[0])
	plt.savefig('.'+root_path+'img/real.png',bbox_inches = 'tight',pad_inches = 0)
	plt.figure()
	plt.axis('off')
	plt.plot(sign[1],'orange')
	plt.savefig('.'+root_path+'img/virtual.png',bbox_inches = 'tight',pad_inches = 0)
	plt.figure()
	plt.axis('off')
	plt.imshow(star)
	plt.savefig('.'+root_path+'img/star.png',bbox_inches = 'tight',pad_inches = 0)
	plt.close()


	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	stars_train = stars_chosse.reshape(-1,1,1024*int(np.sum(obs)))
	datas_train = sign.reshape(-1,1,256)
	train = np.concatenate((stars_train,datas_train),axis =2)
	train = torch.from_numpy(train).float().to(device)


	savemodelpathdict = {1:f'.{root_path}model/stars.pth',3:f'.{root_path}model/stars_haar_reinforcement.pth',7:f'.{root_path}model/stars_haar.pth'}
	savemodelpath = savemodelpathdict[int(sum(obs))]

	model = conv_net.conv_net_harr(11,int(sum(obs))).to(device)

    # 扯淡的路径问题！！！
	sys.path.insert(0, os.getcwd()+root_path)  # 保证Pytorch能够找到模型所在的原始目录
	model = torch.load(savemodelpath)
	model.eval()

	outputs = model(train)
	_, pred = torch.max(outputs, 1)
	
	return labeldic[label],labeldic[int(pred.cpu().numpy())]


	
def RL_dataloader(obs, path):
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
	obs = np.zeros(7)
	obs [0] = 1
	dataharr("none",obs)