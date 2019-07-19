import torch
import torch.nn as nn

class refine_net(nn.Module):

	def __init__(self):
		super(refine_net, self).__init__()

		# 512 x 512
		self.encoder1 = nn.Sequential(
			nn.Conv2d(1, 16, 5, 1, 2),
			nn.BatchNorm2d(16),
			nn.ReLU(True),
			nn.MaxPool2d(2)
			)

		# 256 x 256
		self.encoder2 = nn.Sequential(
			nn.Conv2d(16, 16, 5, 1, 2),
			nn.BatchNorm2d(16),
			nn.ReLU(True),
			nn.MaxPool2d(2)
			)

		# 128 x 128
		self.encoder3 = nn.Sequential(
			nn.Conv2d(16, 16, 5, 1, 2),
			nn.BatchNorm2d(16),
			nn.ReLU(True),
			nn.MaxPool2d(2)
			)

		# 64 x 64
		self.encoder4 = nn.Sequential(
			nn.Conv2d(16, 16, 5, 1, 2),
			nn.BatchNorm2d(16),
			nn.ReLU(True),
			nn.MaxPool2d(2)
			)

		self.rcu = nn.Sequential(
			nn.ReLU(True),
			nn.Conv2d(16, 16, 5, 1, 2),
			)

		self.muti_fusion = nn.Sequential(
			nn.ConvTranspose2d(16, 16, 4, 2, 1),
			nn.BatchNorm2d(16),
			nn.ReLU(True)
			)

		self.final = nn.Sequential(
			nn.ConvTranspose2d(16, 1, 4, 2, 1),
			nn.BatchNorm2d(1),
			nn.Sigmoid()
			)

	def forward(self, img):
		
		e1 = self.encoder1(img)
		e2 = self.encoder2(e1)
		e3 = self.encoder3(e2)
		e4 = self.encoder4(e3)
		
		res = self.rcu(e4) + e4.detach()
		res = self.rcu(e3) + e3.detach() + self.muti_fusion(res).detach()
		res = self.rcu(e2) + e2.detach() + self.muti_fusion(res).detach()
		res = self.rcu(e1) + e1.detach() + self.muti_fusion(res).detach()
		res = self.final(res)

		return res
