import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataloader import ct_dataset
from model import refine_net
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join
import torch.nn.functional as F
import random
import torch.optim as optim
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("-EPOCH", "--EPOCH", dest="epoch", type=int, default=100)
parser.add_argument("-batch_size", dest='batch_size', type=int, default=10)
parser.add_argument("-model", dest='model', type=str, default='refine-net')
parser.add_argument("-lr", dest='lr', type=float, default=1e-4)
args = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if cuda else 'cpu')

if __name__ == '__main__':

	model = refine_net()
	loss_fn = nn.BCELoss()
	dataset = ct_dataset('./dataset/train/data', './dataset/train/label')
	dataset = DataLoader(dataset, batch_size=args.batch_size, shuffle = True)
	optimizer = optim.Adam(model.parameters(), lr = args.lr)
	print(f"""
		Current Hyper-Parameters:
		o Model Type: {args.model},
		o Epoch : {args.epoch},
		o Batch Size : {args.batch_size},
		o Learning Rate : {args.lr},
		""")
	for ep in tqdm(range(args.epoch)):

		for index, batch in enumerate(dataset):
			print(index)
			x, y = batch
			x = x.to(device)
			y = y.to(device)
			
			output = model(x)
			
			loss = loss_fn(output, y)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			test = output[0][0].detach()
			if index == 50:
				plt.imshow(test, cmap='gray')
				plt.show()
