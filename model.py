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
			)

		# 256 x 256
		self.encoder2 = nn.Sequential(
			nn.Conv2d(8, 16, 5, 1, 2),
			nn.BatchNorm2d(16),
			nn.ReLU(True),
			)

		# 128 x 128
		self.encoder3 = nn.Sequential(
			nn.Conv2d(16, 32, 5, 1, 2),
			nn.BatchNorm2d(32),
			nn.ReLU(True),
			)

		# 64 x 64
		self.encoder4 = nn.Sequential(
			nn.Conv2d(32, 64, 5, 1, 2),
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			)

		self.encoder5 = nn.Sequential(
			nn.Conv2d(64, 64, 5, 1, 2),
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			)

		self.encoder6 = nn.Sequential(
			nn.Conv2d(64, 32, 5, 1, 2),
			nn.BatchNorm2d(32),
			nn.ReLU(True),
			)

		self.encoder7 = nn.Sequential(
			nn.Conv2d(32, 16, 5, 1, 2),
			nn.BatchNorm2d(16),
			nn.ReLU(True),
			)

		self.encoder8 = nn.Sequential(
			nn.Conv2d(16, 8, 5, 1, 2),
			nn.BatchNorm2d(8),
			nn.ReLU(True),
			)

		self.encoder9 = nn.Sequential(
			nn.Conv2d(8, 1, 5, 1, 2),
			nn.BatchNorm2d(1),
			nn.ReLU(True),
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
		e5 = self.encoder5(e4)
		e6 = self.encoder6(e5)
		e7 = self.encoder7(e6)
		e8 = self.encoder8(e7)
		e9 = self.encoder9(e8)

		return e9
