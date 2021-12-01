from os import error
from pandas.core.indexes.base import InvalidIndexError
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import loss
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

#time
import time

# Please read the free response questions before starting to code.

def run_model(model,running_mode='train', train_set=None, valid_set=None, test_set=None,
		batch_size=1, learning_rate=0.01, n_epochs=1, stop_thr=1e-4, shuffle=True):
	"""
	This function either trains or evaluates a model.

	training mode: the model is trained and evaluated on a validation set, if provided.
		If no validation set is provided, the training is performed for a fixed
		number of epochs.
		Otherwise, the model should be evaluted on the validation set
		at the end of each epoch and the training should be stopped based on one
		of these two conditions (whichever happens first):
		1. The validation loss stops improving.
		2. The maximum number of epochs is reached.

	testing mode: the trained model is evaluated on the testing set

	Inputs:

	model: the neural network to be trained or evaluated
	running_mode: string, 'train' or 'test'
	train_set: the training dataset object generated using the class MyDataset
	valid_set: the validation dataset object generated using the class MyDataset
	test_set: the testing dataset object generated using the class MyDataset
	batch_size: number of training samples fed to the model at each training step
	learning_rate: determines the step size in moving towards a local minimum
	n_epochs: maximum number of epoch for training the model
	stop_thr: if the validation loss from one epoch to the next is less than this
						value, stop training
	shuffle: determines if the shuffle property of the DataLoader is on/off

	Outputs when running_mode == 'train':

	model: the trained model
	loss: dictionary with keys 'train' and 'valid'
		The value of each key is a list of loss values. Each loss value is the average
		of training/validation loss over one epoch.
		If the validation set is not provided just return an empty list.
	acc: dictionary with keys 'train' and 'valid'
		The value of each key is a list of accuracies (percentage of correctly classified
		samples in the dataset). Each accuracy value is the average of training/validation
		accuracies over one epoch.
		If the validation set is not provided just return an empty list.

	Outputs when running_mode == 'test':

	loss: the average loss value over the testing set.
	accuracy: percentage of correctly classified samples in the testing set.

	Summary of the operations this function should perform:
	1. Use the DataLoader class to generate trainin, validation, or test data loaders
	2. In the training mode:
		- define an optimizer (we use SGD in this homework)
		- call the train function (see below) for a number of epochs untill a stopping
			criterion is met
		- call the test function (see below) with the validation data loader at each epoch
			if the validation set is provided

	3. In the testing mode:
		- call the test function (see below) with the test data loader and return the results

	"""
	#assumptions: training set not None, testing set None, valid set maybe None
	if running_mode == "train":
		#Hopper Dicts
		losses = {"train": [], "valid": []}; accs = {"train": [], "valid": []}
		
		#default assume no valid
		novalid = True
		#data generators
		training_generator = DataLoader(train_set, batch_size, shuffle)
		if valid_set is not None:
			validation_generator = DataLoader(valid_set, batch_size, shuffle)
			novalid = False

		#optimizer with SGD
		optimizer = optim.SGD(model.parameters(), lr=learning_rate)
		
		#run until stopping condition
		
		def met_stopping_condition(epoch, novalid, losses) -> bool:
			#crappy logic but bare with me here
			if novalid:
				#run until end of n_epochs
				return False
			else:
				#dont have any losses to compare yet so skip
				if epoch < 2:
					return False
				#just for dogfc
				elif np.abs(losses["valid"][-2] - losses["valid"][-1]) < stop_thr:
					return True
				#if valid loss stops decreasing, we stop training
				#elif losses["valid"][-1] > losses["valid"][-2]:
				#	return True
				return False

		#time
		#start = time.time()
		epochs_elapsed = 0
		for epoch in range(n_epochs):
			if not met_stopping_condition(epoch, novalid, losses):
				# do work
				model, tr_loss, tr_acc = _train(model, training_generator, optimizer)
				losses["train"].append(tr_loss)
				accs["train"].append(tr_acc)

				#run validation only if we have valid_set
				if valid_set is not None:
					val_loss, val_acc = _test(model, validation_generator)
					losses["valid"].append(val_loss)
					accs["valid"].append(val_acc)
				#update epochs
				epochs_elapsed += 1
			else:
				break
		#end = time.time()
		print(f"Number of epochs ran: {epochs_elapsed}")
		return model, losses, accs
	
	#assuming training set is None, valid test is None, testingset is not None
	elif running_mode == "test":
		testing_generator = DataLoader(test_set, batch_size, shuffle)
		te_loss, te_acc = _test(model, testing_generator, shuffle)
		return te_loss, te_acc

	else:
		print("Invalid mode. Exiting")
		return


def _train(model,data_loader,optimizer,device=torch.device('cpu')):

	"""
	This function implements ONE EPOCH of training a neural network on a given dataset.
	Example: training the Digit_Classifier on the MNIST dataset
	Use nn.CrossEntropyLoss() for the loss function


	Inputs:
	model: the neural network to be trained
	data_loader: for loading the netowrk input and targets from the training dataset
	optimizer: the optimiztion method, e.g., SGD
	device: we run everything on CPU in this homework

	Outputs:
	model: the trained model
	train_loss: average loss value on the entire training dataset
	train_accuracy: average accuracy on the entire training dataset
	"""
	corrects = 0
	total = 0
	running_loss = 0
	loss_criterion = nn.CrossEntropyLoss()

	for local_batch, local_labels in data_loader:
				#stupid line
		local_batch = local_batch.float(); local_labels = local_labels.long()
			
		#send to device (leave out if buggy)
		local_batch, local_labels = local_batch.to(device), local_labels.to(device)

		#zero grads
		optimizer.zero_grad()

		#forward, backward, optimize
		y_hat = model(local_batch)
		loss = loss_criterion(y_hat, local_labels)
		loss.backward()
		optimizer.step()

		#running loss
		running_loss += loss.item() * local_batch.size(0)
		#training acc on training dataset?
		_, predicted = torch.max(y_hat.data, 1)
		corrects += (local_labels == predicted).float().sum()
		total += len(local_labels)

	acc = 100 * corrects/total
	running_loss = running_loss/total
	#smt with the loss tensor needs this additional line
	return model, running_loss, acc


def _test(model, data_loader, device=torch.device('cpu')):
	"""
	This function evaluates a trained neural network on a validation set
	or a testing set.
	Use nn.CrossEntropyLoss() for the loss function

	Inputs:
	model: trained neural network
	data_loader: for loading the netowrk input and targets from the validation or testing dataset
	device: we run everything on CPU in this homework

	Output:
	test_loss: average loss value on the entire validation or testing dataset
	test_accuracy: percentage of correctly classified samples in the validation or testing dataset
	"""
	loss_criterion = nn.CrossEntropyLoss()
	correct = 0
	total = 0
	test_loss = 0
	# since we're not training, we don't need to calculate the gradients for our outputs
	with torch.no_grad():
		for data in data_loader:
			images, labels = data
			images = images.float(); labels = labels.long()
			# calculate outputs by running images through the network
			outputs = model(images)

			#loss											#life saver
			test_loss += loss_criterion(outputs, labels).item() * images.size(0)

			# the class with the highest energy is what we choose as prediction
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	test_loss /= len(data_loader.dataset)

	test_acc = (100 * correct / total)
	return test_loss, test_acc

