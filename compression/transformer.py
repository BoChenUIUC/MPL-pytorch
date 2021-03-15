import cv2
import numpy as np
import time
import torch
import glob

from torchvision import transforms
from torch.utils.data import Dataset
from collections import OrderedDict
from PIL import Image
from io import StringIO
import pickle,sys,os
import subprocess
import matplotlib.pyplot as plt
import matplotlib

dataset = 'ucf101-24'

def get_edge_feature(frame, edge_blur_rad=11, edge_blur_var=0, edge_canny_low=101, edge_canny_high=255):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (edge_blur_rad, edge_blur_rad), edge_blur_var)
	edge = cv2.Canny(blur, edge_canny_low, edge_canny_high)
	return edge
    

def get_KAZE_feature(frame):
	alg = cv2.KAZE_create()
	kps = alg.detect(frame)
	kps = sorted(kps, key=lambda x: -x.response)[:32]
	points = [p.pt for p in kps]
	return points

def get_harris_corner(frame):
	img = frame.copy()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	dst = cv2.cornerHarris(gray,2,3,0.04)
	# Threshold for an optimal value, it may vary depending on the image.
	dst[dst>0.01*dst.max()]=[255]
	dst[dst<255]=[0]
	return dst

def get_GFTT(frame):
	img = frame.copy()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
	if corners is not None:
		corners = np.int0(corners) 
		points = [i.ravel() for i in corners]
	else:
		points = []
	return points

# pip install opencv-python==3.4.2.16
# pip install opencv-contrib-python==3.4.2.16
def get_SIFT(frame):
	img = frame.copy()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	sift = cv2.xfeatures2d.SIFT_create()
	kps = sift.detect(gray,None)
	points = [p.pt for p in kps]
	return points

def get_SURF(frame):
	img = frame.copy()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	surf = cv2.xfeatures2d.SURF_create()
	kps = surf.detect(gray,None)
	points = [p.pt for p in kps]
	return points

def get_FAST(frame):
	img = frame.copy()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	fast = cv2.FastFeatureDetector_create(threshold=50)
	kps = fast.detect(img,None)
	points = [p.pt for p in kps]
	return points

def get_STAR(frame):
	img = frame.copy()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# Initiate STAR detector
	star = cv2.xfeatures2d.StarDetector_create()

	# find the keypoints with STAR
	kps = star.detect(img,None)
	points = [p.pt for p in kps]
	return points

def get_ORB(frame):
	img = frame.copy()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	orb = cv2.ORB_create()
	kps = orb.detect(img,None)
	points = [p.pt for p in kps]
	return points

def count_point_in_ROI(ROI, points):
	counter = 0
	for px,py in points:
		inROI = False
		x1,y1,x2,y2 = ROI
		if x1<=px and x2>px and y1<=py and y2>py:
			inROI = True
		if inROI:counter += 1
	return counter

def count_mask_in_ROI(ROI, mp):
	total_pts = np.count_nonzero(mp)
	x1,y1,x2,y2 = ROI
	mp[y1:y2,x1:x2] = 0
	nonroi_pts = np.count_nonzero(mp)
	return total_pts-nonroi_pts

def ROI_area(ROIs,w,h):
	im = np.zeros((h,w),dtype=np.uint8)
	for x1,y1,x2,y2 in ROIs:
		im[y1:y2,x1:x2] = 1
	roi = np.count_nonzero(im)
	return roi

def path_to_disturbed_image(pil_image, label, r_in, r_out):
	b,g,r = cv2.split(np.array(pil_image))
	np_img = cv2.merge((b,g,r))
	np_img = region_disturber(np_img,label, r_in, r_out)
	pil_image = Image.fromarray(np_img)
	return pil_image

# change quality of non-ROI
# r_in is the scaled ratio of ROIs
# r_out is the scaled ratio of the whole image
def region_disturber(image,label,r_in,r_out):
	# get the original content from ROI
	# downsample rest, then upsample
	# put roi back
	w,h = 320,240
	dsize_out = (int(w*r_out),int(h*r_out))
	crops = []
	for _,cx,cy,imgw,imgh  in label:
		cx=int(cx*320);cy=int(cy*240);imgw=int(imgw*320);imgh=int(imgh*320)
		x1=max(cx-imgw//2,0);x2=min(cx+imgw//2,w);y1=max(cy-imgw//2,0);y2=min(cy+imgw//2,h)
		crop = image[y1:y2,x1:x2]
		if r_in<1:
			dsize_in = (int((x2-x1)*r_in),int((y2-y1)*r_in))
			crop_d = cv2.resize(crop, dsize=dsize_in, interpolation=cv2.INTER_LINEAR)
			crop = cv2.resize(crop_d, dsize=(x2-x1,y2-y1), interpolation=cv2.INTER_LINEAR)
		crops.append((x1,y1,x2,y2,crop))
	if r_out<1:
		# downsample
		downsample = cv2.resize(image, dsize=dsize_out, interpolation=cv2.INTER_LINEAR)
		# upsample
		image = cv2.resize(downsample, dsize=(w,h), interpolation=cv2.INTER_LINEAR)
	for x1,y1,x2,y2,crop  in crops:
		image[y1:y2,x1:x2] = crop
	
	return image

# analyze static and motion feature points
# need to count the number of features ROI and not in ROI
# calculate the density
# should compare  
# percentage of features/percentage of area
def analyzer(image):
	# analyze features in image
	bgr_frame = np.array(image)
	# FAST
	fast, _ = get_FAST(bgr_frame)
	# STAR
	star, _ = get_STAR(bgr_frame)
	# ORB
	orb, _ = get_ORB(bgr_frame)

class LRU(OrderedDict):

	def __init__(self, maxsize=128):
		self.maxsize = maxsize
		super().__init__()

	def __getitem__(self, key):
		value = super().__getitem__(key)
		self.move_to_end(key)
		return value

	def __setitem__(self, key, value):
		if key in self:
			self.move_to_end(key)
		super().__setitem__(key, value)
		if len(self) > self.maxsize:
			oldest = next(iter(self))
			del self[oldest]

def heatmap(data, row_labels, col_labels, ax=None,
			cbar_kw={}, cbarlabel="", **kwargs):
	if not ax:
		ax = plt.gca()

	# Plot the heatmap
	im = ax.imshow(data, **kwargs)

	# Create colorbar
	cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
	cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

	# We want to show all ticks...
	ax.set_xticks(np.arange(data.shape[1]))
	ax.set_yticks(np.arange(data.shape[0]))
	# ... and label them with the respective list entries.
	ax.set_xticklabels(col_labels)
	ax.set_yticklabels(row_labels)

	# Let the horizontal axes labeling appear on top.
	ax.tick_params(top=True, bottom=False,
					labeltop=True, labelbottom=False)

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
					rotation_mode="anchor")

	# Turn spines off and create white grid.
	for edge, spine in ax.spines.items():
		spine.set_visible(False)

	ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
	ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
	ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
	ax.tick_params(which="minor", bottom=False, left=False)

	return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
					 textcolors=("black", "white"),
					 threshold=None, **textkw):

	if not isinstance(data, (list, np.ndarray)):
		data = im.get_array()

	# Normalize the threshold to the images color range.
	if threshold is not None:
		threshold = im.norm(threshold)
	else:
		threshold = im.norm(data.max())/2.

	# Set default alignment to center, but allow it to be
	# overwritten by textkw.
	kw = dict(horizontalalignment="center",
			verticalalignment="center")
	kw.update(textkw)

	# Get the formatter in case a string is supplied
	if isinstance(valfmt, str):
		valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

	# Loop over the data and create a `Text` for each "pixel".
	# Change the text's color depending on the data.
	texts = []
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
			text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
			texts.append(text)

	return texts

def tile_encoder(image, C_param, counter, snapshot=False):
	start = time.perf_counter()
	toSave = snapshot and counter<5
	# analyze features in image
	bgr_frame = np.array(image)
	if toSave:
		cv2.imwrite(f'samples/{counter:2}_origin.jpg',bgr_frame)
	# harris corner
	# hc, _ = get_harris_corner(bgr_frame)
	# FAST
	fast = get_FAST(bgr_frame)
	# STAR
	star = get_STAR(bgr_frame)
	# ORB
	orb = get_ORB(bgr_frame)

	point_features = [fast, star, orb]
	map_features = []
	num_features = len(point_features) + len(map_features)
	# snapshot features optionally
	if toSave:
		feature_frame = np.zeros(bgr_frame.shape)
		colors = [(72, 31, 219),(112, 70, 28),(168, 182, 33)]
		for points,color in zip(point_features,colors):
			for px,py in points:
				feature_frame = cv2.circle(feature_frame, (int(px),int(py)), radius=2, color=color, thickness=-1)
		cv2.imwrite(f'samples/{counter:2}_feature.jpg',feature_frame)
	# divide image to 4*3 tiles
	ROIs = []
	num_w, num_h = 4,3
	img_h,img_w = bgr_frame.shape[:2]
	tilew,tileh = img_w//num_w,img_h//num_h
	if img_w%num_w != 0:tilew += 1
	if img_h%num_h != 0:tileh += 1
	for row in range(num_h):
		for col in range(num_w):
			x1 = col*tilew; x2 = min((col+1)*tilew,img_w); y1 = row*tileh; y2 = min((row+1)*tileh,img_h)
			ROIs.append([x1,y1,x2,y2])
	counts = np.zeros((num_w*num_h,num_features))
	for roi_idx,ROI in enumerate(ROIs):
		feat_idx = 0
		for mf in map_features:
			c = count_mask_in_ROI(ROI,mf)
			counts[roi_idx,feat_idx] = c
			feat_idx += 1
		for pf in point_features:
			c = count_point_in_ROI(ROI,pf)
			counts[roi_idx,feat_idx] = c
			feat_idx += 1

	# weight of different features
	weights = C_param[:num_features] + 0.5
	# lower and upper
	lower,upper = C_param[num_features:num_features+2] + 0.5
	lower,upper = min(lower,upper),max(lower,upper)
	lower = max(lower,0);upper = min(upper,1)
	# order to adjust the concentration of the  scores
	k = int(((C_param[num_features+2]+0.5)*5))
	k = min(k,4); k = max(k,0)
	order_choices = [1./3,1./2,1,2,3]
	# score of each feature sum to 1
	normalized_score = counts/(np.sum(counts,axis=0)+1e-6)
	weights /= (np.sum(weights)+1e-6)
	# ws of all tiles sum up to 1
	weighted_scores = np.matmul(normalized_score,weights)
	# the weight is more valuable when its value is higher
	quality = (upper-lower)*weighted_scores**order_choices[k] + lower
	# generate heatmap
	if toSave:
		hm = np.reshape(quality,(num_h,num_w))
		# plt.imshow(hm, cmap='hot', interpolation='nearest')
		# plt.savefig(f'samples/{counter:2}_heatmap.jpg')
		fig, ax = plt.subplots()

		im, cbar = heatmap(hm, [str(i) for i in range(num_h)],
						 [str(i) for i in range(num_w)], ax=ax,
		                   cmap="coolwarm", cbarlabel="Down-sampling rate")
		texts = annotate_heatmap(im, valfmt="{x:.2f}")

		fig.tight_layout()
		plt.savefig(f'samples/{counter:2}_heatmap.jpg')

	tile_sizes = [(int(np.rint((x2-x1)*r)),int(np.rint((y2-y1)*r))) for r,(x1,y1,x2,y2) in zip(quality,ROIs)]

	# not used for training,but can be used for 
	# ploting the pareto front
	compressed_size = 0
	original_size = 0
	tile_size = tilew * tileh
	for roi,dsize in zip(ROIs,tile_sizes):
		x1,y1,x2,y2 = roi
		if dsize == (x2-x1,y2-y1):
			compressed_size += (x2-x1)*(y2-y1)
			continue
		crop = bgr_frame[y1:y2,x1:x2].copy()
		if dsize[0]==0 or dsize[1]==0:
			bgr_frame[y1:y2,x1:x2] = [0]
		else:
			try:
				original_size += len(pickle.dumps(crop, 0))
				crop_d = cv2.resize(crop, dsize=dsize, interpolation=cv2.INTER_LINEAR)
				compressed_size += len(pickle.dumps(crop_d, 0))
				crop = cv2.resize(crop_d, dsize=(x2-x1,y2-y1), interpolation=cv2.INTER_LINEAR)
			except Exception as e:
				print(repr(e))
				print(C_param,tile_sizes)
				print('dsize:',dsize,crop.shape)
				exit(1)
			bgr_frame[y1:y2,x1:x2] = crop

	end = time.perf_counter()
	if toSave:
		cv2.imwrite(f'samples/{counter:2}_compressed.jpg',bgr_frame)
	return bgr_frame,original_size,compressed_size,end-start

def advanced_tile_encoder(image, C_param, counter, alg='WebP', snapshot=False):
	start = time.perf_counter()
	toSave = snapshot and counter<5
	# analyze features in image
	bgr_frame = np.array(image)
	if toSave:
		cv2.imwrite(f'samples/{counter:2}_origin.jpg',bgr_frame)
	# harris corner
	# hc, _ = get_harris_corner(bgr_frame)
	# FAST
	fast = get_FAST(bgr_frame)
	# STAR
	star = get_STAR(bgr_frame)
	# ORB
	orb = get_ORB(bgr_frame)

	point_features = [fast, star, orb]
	map_features = []
	num_features = len(point_features) + len(map_features)
	# snapshot features optionally
	if toSave:
		feature_frame = np.zeros(bgr_frame.shape)
		colors = [(72, 31, 219),(112, 70, 28),(168, 182, 33)]
		for points,color in zip(point_features,colors):
			for px,py in points:
				feature_frame = cv2.circle(feature_frame, (int(px),int(py)), radius=2, color=color, thickness=-1)
		cv2.imwrite(f'samples/{counter:2}_feature.jpg',feature_frame)
	# divide image to 4*3 tiles
	ROIs = []
	num_w, num_h = 4,3
	img_h,img_w = bgr_frame.shape[:2]
	tilew,tileh = img_w//num_w,img_h//num_h
	if img_w%num_w != 0:tilew += 1
	if img_h%num_h != 0:tileh += 1
	for row in range(num_h):
		for col in range(num_w):
			x1 = col*tilew; x2 = min((col+1)*tilew,img_w); y1 = row*tileh; y2 = min((row+1)*tileh,img_h)
			ROIs.append([x1,y1,x2,y2])
	counts = np.zeros((num_w*num_h,num_features))
	for roi_idx,ROI in enumerate(ROIs):
		feat_idx = 0
		for mf in map_features:
			c = count_mask_in_ROI(ROI,mf)
			counts[roi_idx,feat_idx] = c
			feat_idx += 1
		for pf in point_features:
			c = count_point_in_ROI(ROI,pf)
			counts[roi_idx,feat_idx] = c
			feat_idx += 1

	# weight of different features
	weights = C_param[:num_features] + 0.5
	# lower and upper
	lower,upper = C_param[num_features:num_features+2] + 0.5
	lower,upper = min(lower,upper),max(lower,upper)
	lower = max(lower,0);upper = min(upper,1)
	# order to adjust the concentration of the  scores
	k = int(((C_param[num_features+2]+0.5)*5))
	k = min(k,4); k = max(k,0)
	order_choices = [1./3,1./2,1,2,3]
	# score of each feature sum to 1
	normalized_score = counts/(np.sum(counts,axis=0)+1e-6)
	weights /= (np.sum(weights)+1e-6)
	# ws of all tiles sum up to 1
	weighted_scores = np.matmul(normalized_score,weights)
	# the weight is more valuable when its value is higher
	quality = (upper-lower)*weighted_scores**order_choices[k] + lower
	# generate heatmap
	if toSave:
		hm = np.reshape(quality,(num_h,num_w))
		# plt.imshow(hm, cmap='hot', interpolation='nearest')
		# plt.savefig(f'samples/{counter:2}_heatmap.jpg')
		fig, ax = plt.subplots()

		im, cbar = heatmap(hm, [str(i) for i in range(num_h)],
						 [str(i) for i in range(num_w)], ax=ax,
		                   cmap="coolwarm", cbarlabel="Down-sampling rate")
		texts = annotate_heatmap(im, valfmt="{x:.2f}")

		fig.tight_layout()
		plt.savefig(f'samples/{counter:2}_heatmap.jpg')

	tile_sizes = [(int(np.rint((x2-x1)*r)),int(np.rint((y2-y1)*r))) for r,(x1,y1,x2,y2) in zip(quality,ROIs)]

	# not used for training,but can be used for 
	# ploting the pareto front
	compressed_size = 0
	original_size = 0
	for roi,r in zip(ROIs,quality):
		x1,y1,x2,y2 = roi
		crop = bgr_frame[y1:y2,x1:x2].copy()
		r *= 100
		try:
			if alg == 'WebP':
				crop,osize,csize,_ = WebP(crop,r)
			elif alg == 'JPEG':
				crop,osize,csize,_ = JPEG(crop,r)
			else:
				print('Algoritm not supported.')
				exit(1)
		except Exception as e:
			print(repr(e))
			print(r)
			print(crop.shape)
			exit(1)
		original_size += osize
		compressed_size += csize
		bgr_frame[y1:y2,x1:x2] = crop

	end = time.perf_counter()
	if toSave:
		cv2.imwrite(f'samples/{counter:2}_compressed.jpg',bgr_frame)
	return bgr_frame,original_size,compressed_size,end-start

def JPEG2000(npimg,C_param):
	comp_dir = 'compression/jpeg2000/'
	tmp_dir = comp_dir + 'tmp/'
	cv2.imwrite(tmp_dir+'origin.png',npimg)
	osize = os.stat(tmp_dir+'origin.png').st_size
	start = time.perf_counter()
	comp_cmd = './'+comp_dir+'opj_compress -i '+tmp_dir+'origin.png -o '+tmp_dir+'compressed.j2k -r '+str(C_param)
	subprocess.call(comp_cmd, shell=True)
	decm_cmd = './'+comp_dir+'opj_decompress -i '+tmp_dir+'compressed.j2k -o '+tmp_dir+'decompressed.png -r '+str(C_param)
	subprocess.call(decm_cmd, shell=True)
	end = time.perf_counter()
	lossy_image = cv2.imread(tmp_dir+'decompressed.png')
	assert(lossy_image is not None)
	csize = os.stat(tmp_dir+'compressed.j2k').st_size
	return lossy_image,osize,csize,end-start

def JPEG(npimg,C_param):
	start = time.perf_counter()
	encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), C_param]
	osize = len(pickle.dumps(npimg, 0))
	result, lossy_image = cv2.imencode('.jpg', npimg, encode_param)
	csize = len(pickle.dumps(lossy_image, 0))
	lossy_image = cv2.imdecode(lossy_image, cv2.IMREAD_COLOR)
	end = time.perf_counter()
	return lossy_image,osize,csize,end-start

def WebP(npimg,C_param):
	encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), C_param]
	osize = len(pickle.dumps(npimg, 0))
	start = time.perf_counter()
	result, lossy_image = cv2.imencode('.webp', npimg, encode_param)
	end = time.perf_counter()
	csize = len(pickle.dumps(lossy_image, 0))
	lossy_image = cv2.imdecode(lossy_image, cv2.IMREAD_COLOR)
	return lossy_image,osize,csize,end-start

# define a class for transformation
class Transformer:
	def __init__(self,name,snapshot = False):
		# need a dict as buffer to store transformed image of a range
		self.name = name
		self.snapshot = snapshot
		self.counter = 0
		self.time = []

	def transform(self, image=None, C_param=None):
		# need to recover images and print examples
		# get JPEG lib
		if self.name == 'JPEG':
			# 0->100
			rimage,osize,csize,t = JPEG(image,C_param)
		elif self.name == 'JPEG2000':
			rimage,osize,csize,t = JPEG2000(image,C_param)
		elif self.name == 'WebP':
			# 1-100
			rimage,osize,csize,t = WebP(image,C_param)
		elif self.name == 'TiledWebP' or self.name == 'TiledJPEG':
			rimage,osize,csize,t = advanced_tile_encoder(image, C_param, self.counter, self.name[5:], self.snapshot)
		elif self.name == 'Tiled':	
			rimage,osize,csize,t = tile_encoder(image, C_param, self.counter, self.snapshot)
		else:
			print(self.name,'not implemented.')
			exit(1)
		self.original_size += osize
		self.compressed_size += csize
		self.counter += 1
		self.time += [t]
		return rimage

	def reset(self):
		self.compressed_size = 0
		self.original_size = 0

	def get_compression_ratio(self):
		assert(self.original_size>0)
		return 1-1.0*self.compressed_size/self.original_size

	def get_compression_time(self):
		return np.mean(self.time),np.std(self.time)

def test_dataloader():
	from torchvision import datasets
	from torchvision import transforms
	from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
	transform_val = transforms.Compose([transforms.ToTensor(),])
	test_dataset = datasets.CIFAR10('.././data', train=True, 
					transform=transform_val, download=False)
	test_loader = DataLoader(test_dataset,
					sampler=SequentialSampler(test_dataset),
					batch_size=64,
					num_workers=4)
	with torch.no_grad():
		for step, (images, targets) in enumerate(test_loader):
			for th_img,target in zip(images,targets):
				npimg = (th_img.permute(1,2,0).numpy()*255).astype(np.uint8)
				pngimg,pngosize,pngcsize = PNG(npimg,5)
				print(pngosize,pngcsize)
				jpegimg,jpegosize,jpegcsize = JPEG(npimg,50)
				print(jpegosize,jpegcsize)
				jp2img,jp2isize,jp2csize = JPEG2000(npimg,0)
				print(jp2isize,jp2csize)

				cv2.imwrite('0.png',npimg)
				cv2.imwrite('1.png',pngimg)
				cv2.imwrite('2.png',jpegimg)
				cv2.imwrite('3.png',jp2img)
				exit(0)

if __name__ == "__main__":
    # img = cv2.imread('/home/bo/research/dataset/ucf24/compressed/000000.jpg')
    # img = cv2.imread('/home/bo/research/dataset/ucf24/rgb-images/Basketball/v_Basketball_g01_c01/00001.jpg')
    # analyzer(img)
    # _,osize,csize = JPEG(img,0)
    # PNG(img,9)
    # _,osize,csize = JPEG2000(img,100)
    # print(osize,csize)
    test_dataloader()