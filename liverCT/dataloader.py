from torch.utils.data import Dataset, DataLoader
import os
from os.path import join
from os import listdir
from PIL import Image
from torchvision import transforms
import torch
import matplotlib.pyplot as plt

std = [0.5]
mean = [0.5]

transform = transforms.Compose([
	# transforms.Grayscale(),
	transforms.ToTensor(),
	transforms.Normalize(mean, std)
	])

class ct_dataset(Dataset):

	def __init__(self, image_path, label_path):

		self.image_path = image_path
		self.label_path = label_path

		self.images = [file for file in listdir(join(os.getcwd(), image_path)) if file != '.DS_Store']
		self.labels = [file for file in listdir(join(os.getcwd(), label_path)) if file != '/DS_Store']
		self.images = sorted(self.images, key=lambda x:int(x))


		self.total_images = []
		for index, directory in enumerate(self.images):
			num = listdir(join(os.getcwd(), image_path, directory))
			self.total_images.append(len(num))

	def __len__(self):
		return sum(self.total_images)


	def __getitem__(self, idx):

		directory_name = len(self.total_images)-1
		for index, num in enumerate(self.total_images):
			if idx >= num:
				idx -= num
			else:
				directory_name = index
				break

		directory_name = str(directory_name)

		filename = listdir(join(os.getcwd(), self.image_path, directory_name))[idx]
		
		image = Image.open(join(os.getcwd(), self.image_path, directory_name, filename))
		label = Image.open(join(os.getcwd(), self.label_path, directory_name, filename))

		image = transform(image)
		label =transform(label)
		
		return image, label
		
# if __name__ == '__main__':

# 	dataset = ct_dataset('./dataset/train/data', './dataset/train/label')
# 	dataset = DataLoader(dataset, batch_size=1)
# 	for i in dataset:
# 		pass