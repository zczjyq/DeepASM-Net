import os
import numpy as np
import cv2


class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


class ListAverageMeter(object):
	"""Computes and stores the average and current values of a list"""
	def __init__(self):
		self.len = 10000  # set up the maximum length
		self.reset()

	def reset(self):
		self.val = [0] * self.len
		self.avg = [0] * self.len
		self.sum = [0] * self.len
		self.count = 0

	def set_len(self, n):
		self.len = n
		self.reset()

	def update(self, vals, n=1):
		assert len(vals) == self.len, 'length of vals not equal to self.len'
		self.val = vals
		for i in range(self.len):
			self.sum[i] += self.val[i] * n
		self.count += n
		for i in range(self.len):
			self.avg[i] = self.sum[i] / self.count
			

def read_img(filename):
	# 使用 np.fromfile + imdecode 支持中文路径（cv2.imread 在 Windows 下对中文路径有问题）
	img_buf = np.fromfile(filename, dtype=np.uint8)
	img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
	if img is None:
		raise FileNotFoundError(f"无法读取图像: {filename}")
	return img[:, :, ::-1].astype('float32') / 255.0


def write_img(filename, img):
	# 使用 imencode + tofile 支持中文路径
	img_bgr = np.round((img[:, :, ::-1].copy() * 255.0)).astype('uint8')
	ext = os.path.splitext(filename)[1] or '.png'
	_, img_buf = cv2.imencode(ext, img_bgr)
	img_buf.tofile(filename)


def hwc_to_chw(img):
	return np.transpose(img, axes=[2, 0, 1]).copy()


def chw_to_hwc(img):
	return np.transpose(img, axes=[1, 2, 0]).copy()
