import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import join
import cv2
import os

def _load_itk(filename):

	itkImage = sitk.ReadImage(filename)
	ct_scan = sitk.GetArrayFromImage(itkImage)
	origin = np.array(list(reversed(itkImage.GetOrigin())))
	spacing = np.array(list(reversed(itkImage.GetSpacing())))

	return ct_scan, origin, spacing


def create_dataset():
	root = ['scan','label']

	if not os.path.isdir('./dataset'):
		os.mkdir('./dataset')
		os.mkdir('./dataset/train')
		os.mkdir('./dataset/train/data')
		os.mkdir('./dataset/train/label')
		os.mkdir('./dataset/test')
		os.mkdir('./dataset/test/data')
		os.mkdir('./dataset/test/label')
		
	for name in root:
		files = listdir(join('raw_data', name))
		valid_files = [file for file in files if file.endswith('.mhd')]
		
		for index, mhd in enumerate(valid_files):
			data, _ , _ = _load_itk(join('raw_data',name, mhd))
			print('File [%d/%d] is Complete' % (index, len(valid_files)))
			if name == 'scan':
				if not os.path.isdir('./dataset/train/data/'+str(index)):
					os.mkdir('./dataset/train/data/'+str(index))
			elif name == 'label':
				if not os.path.isdir('./dataset/train/label/'+str(index)):
					os.mkdir('./dataset/train/label/'+str(index))

			for img_idx, img in enumerate(data):
				
				save_name = "{0:3}".format(img_idx).replace(' ', "0")
				if name == 'scan':
					cv2.imwrite('./dataset/train/data/'+str(index)+'/'+save_name+'.jpg', img)
				elif name == 'label':
					cv2.imwrite('./dataset/train/label/'+str(index)+'/'+save_name+'.jpg', img*255)

if __name__ == '__main__':
	create_dataset()
	