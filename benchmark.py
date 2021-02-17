import config
from dataset import FaceData
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
from model import load_model

images_root = os.path.join(config.data_root, 'Images')


def benchmark(model, valdir):
    val_data = FaceData(images_root, valdir, config.transform)
    val_loader = DataLoader(val_data, batch_size=16)

    with torch.no_grad():
        label = []
        pred = []
        for i, (img, target) in enumerate(val_loader):
            img, target = img.to(torch.device(config.device)), target.to(torch.device(config.device))
            output = model(img).squeeze(1)
            label.append(target.cpu()[0])
            pred.append(output.cpu()[0])

        # measurements
        label = np.array(label)
        pred = np.array(pred)
        correlation = np.corrcoef(label, pred)[0][1]
        mae = np.mean(np.abs(label - pred))
        rmse = np.sqrt(np.mean(np.square(label - pred)))

    print('Correlation:{correlation:.4f}\t'
        'Mae:{mae:.4f}\t'
        'Rmse:{rmse:.4f}\t'.format(
            correlation=correlation, mae=mae, rmse=rmse))
    return correlation, mae, rmse


if __name__ == '__main__':
	# The split of 60% training and 40% testing
	saved_path = os.path.join(config.models_dir, config.model_arch+'_best_state.pt')
	resnet_model = load_model(config.model_arch)
	resnet_model.load_state_dict(torch.load(saved_path, map_location=torch.device(config.device)))
	resnet_model.eval()
	val_dir = os.path.join(config.data_root, 'train_test_files/split_of_60%training and 40%testing/test.txt')
	correlation, mae, rmse = benchmark(resnet_model, val_dir)