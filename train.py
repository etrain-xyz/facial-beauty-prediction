import config
import copy
import numpy as np
from model import load_model
import torch.nn as nn
from dataset import FaceData
from torch.utils.data import DataLoader
from optimizer import Adas
import torch
import os

# if the distance of pred and groundtruth is smaller than error_tolerance, we regard the pred as a right one.
error_tolerance = 0.5

criterion = nn.MSELoss()

images_root = os.path.join(config.data_root, 'Images')

def train(train_dir, val_dir, model_saved_path):
	epochs = config.epochs
	model = load_model(config.model_arch)
	optimizer = Adas(model.parameters())
	train_data = FaceData(images_root, train_dir, config.transform)
	train_loader = DataLoader(train_data, batch_size=16)
	val_data = FaceData(images_root, val_dir, config.transform)
	val_loader = DataLoader(val_data, batch_size=16)
	train_losses = []
	valid_losses = []
	best_acc = 0.0
	best_model_wts = copy.deepcopy(model.state_dict())

	for epoch in range(1, epochs+1):
		# Train Process
		model.train()
		batch_losses=[]
		for i, data in enumerate(train_loader):
			inputs, labels = data
			inputs, labels = inputs.to(torch.device(config.device)), labels.to(torch.device(config.device))
			optimizer.zero_grad()
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			batch_losses.append(loss.item())
		
		train_losses.append(batch_losses)
		print(f'Epoch - {epoch} Train-Loss : {np.mean(train_losses[-1])}')

		# Valid Process
		model.eval()
		batch_losses=[]
		n_correct = 0
		n_total = 0
		for i, data in enumerate(val_loader):
			inputs, labels = data
			inputs, labels = inputs.to(torch.device(config.device)), labels.to(torch.device(config.device))
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			batch_losses.append(loss.item())
			n_correct += (abs(labels - outputs) < error_tolerance).sum().item()
			n_total += inputs.size(0)
		valid_losses.append(batch_losses)
		accuracy = n_correct / n_total
		print(f'Epoch - {epoch} Valid-Loss : {np.mean(valid_losses[-1])} Valid-Accuracy : {accuracy}')

		# deep copy the model
		if accuracy > best_acc:
			best_acc = accuracy
			best_model_wts = copy.deepcopy(model.state_dict())

	# load best model weights
	model.load_state_dict(best_model_wts)
	torch.save(model.state_dict(), model_saved_path)


if __name__ == '__main__':
	# The split of 60% training and 40% testing
	train_dir = os.path.join(config.data_root, 'train_test_files/split_of_60%training and 40%testing/train.txt')
	val_dir = os.path.join(config.data_root, 'train_test_files/split_of_60%training and 40%testing/test.txt')
	saved_path = os.path.join(config.models_dir, config.model_arch+'_best_state.pt')
	train(train_dir, val_dir, model_saved_path=saved_path)