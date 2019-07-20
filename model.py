import torch
import torch.nn as nn

class refine_net(nn.Module):

	def __init__(self):
		super(refine_net, self).__init__()

		# 512 x 512
		self.encoder1 = nn.Sequential(
			nn.Conv2d(1, 8, 5, 1, 2),
			nn.BatchNorm2d(8),
			nn.ReLU(True),
			nn.MaxPool2d(2)
			)

		# 256 x 256
		self.encoder2 = nn.Sequential(
			nn.Conv2d(8, 16, 5, 1, 2),
			nn.BatchNorm2d(16),
			nn.ReLU(True),
			nn.MaxPool2d(2)
			)

		# 128 x 128
		self.encoder3 = nn.Sequential(
			nn.Conv2d(16, 32, 5, 1, 2),
			nn.BatchNorm2d(32),
			nn.ReLU(True),
			nn.MaxPool2d(2)
			)

		# 64 x 64
		self.encoder4 = nn.Sequential(
			nn.Conv2d(32, 64, 5, 1, 2),
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			nn.MaxPool2d(2)
			)

		self.rcu4 = nn.Sequential(
			nn.ReLU(True),
			nn.Conv2d(64, 64, 5, 1, 2),
			)

		self.rcu3 = nn.Sequential(
			nn.ReLU(True),
			nn.Conv2d(32, 32, 5, 1, 2),
			)
		
		self.rcu2 = nn.Sequential(
			nn.ReLU(True),
			nn.Conv2d(16, 16, 5, 1, 2),
			)
		
		self.rcu1 = nn.Sequential(
			nn.ReLU(True),
			nn.Conv2d(8, 8, 5, 1, 2),
			)
		

		self.muti_fusion3 = nn.Sequential(
			nn.ConvTranspose2d(64, 32, 4, 2, 1),
			nn.BatchNorm2d(32),
			nn.ReLU(True)
			)

		self.muti_fusion2 = nn.Sequential(
			nn.ConvTranspose2d(32, 16, 4, 2, 1),
			nn.BatchNorm2d(16),
			nn.ReLU(True)
			)

		self.muti_fusion1 = nn.Sequential(
			nn.ConvTranspose2d(16, 8, 4, 2, 1),
			nn.BatchNorm2d(8),
			nn.ReLU(True)
			)

		
		self.final = nn.Sequential(
			nn.ConvTranspose2d(8, 1, 4, 2, 1),
			nn.BatchNorm2d(1),
			nn.Sigmoid()
			)


	def forward(self, img):
		# 8 x 256 x 256
		e1 = self.encoder1(img)
		# 16 x 128 x 128
		e2 = self.encoder2(e1)
		# 32 x 64 x 64
		e3 = self.encoder3(e2)
		# 64 x 32 x 32
		e4 = self.encoder4(e3)
		
		# 64 x 32 x 32
		res = self.rcu4(e4) + e4.detach()
		# 32 x 64 x 64
		res = self.rcu3(e3) + e3.detach() + self.muti_fusion3(res).detach()
		# 16 x 128 x 128
		res = self.rcu2(e2) + e2.detach() + self.muti_fusion2(res).detach()
		# 8 x 256 x 256
		res = self.rcu1(e1) + e1.detach() + self.muti_fusion1(res).detach()
		
		res = self.final(res)

		return res
