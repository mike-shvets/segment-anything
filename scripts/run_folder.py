import argparse
import os
import sys
import cv2
from tqdm import tqdm
import numpy as np
from natsort import natsorted
from glob import glob
from time import time

# SAM
from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator

# Mask plotting
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def reproject():
	pass


def add_mask_to_canvas(mask, ax, color):
	color = np.concatenate([color, np.array([0.6])], axis=0)
	h, w = mask.shape[-2:]
	mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
	ax.imshow(mask_image)


def read_mesh(path):
	assert path.endswith(".pth")
	locs, feats, labels = torch.load(path)
	labels = labels.astype("uint8")
	feats = (feats + 1.) * 127.5
	return locs, feats, labels


def read_pose(path):
	pose = np.asarray(
	    [[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in
	     (x.split(" ") for x in open(path).read().splitlines())]
	)
	return pose


def get_sample_name(path):
	filename = os.path.split(path)[-1]
	return os.path.splitext(filename)[0]


def get_paths(folder, names, ext):
	paths = [os.path.join(folder, name + ext) for name in names]
	return paths


def filter_poses(paths):
	filtered = list()
	for path in paths:
		pose = read_pose(path)
		if np.isfinite(pose).all():
			filtered.append(path)
	return filtered


if __name__ == "__main__":
	parser = argparse.ArgumentParser("Args")
	parser.add_argument("-r", "--rgb_folder", required=True,
						help="Path to folder with RGB images")
	parser.add_argument("-d", "--depth_folder",
						help="Path to folder with depth maps")
	parser.add_argument("-p", "--poses_folder",
						help="Path to folder with camera poses")
	parser.add_argument("-m", "--mesh_file",
						help="Path to the mesh file (.pth)")
	parser.add_argument("-o", "--output_folder", default="./tmp_output",
						help="Ouput folder")
	parser.add_argument("-c", "--checkpoint_path",
						default="../checkpoints/sam_vit_h_4b8939.pth",
						help="Path to the SAM checkpoint")
	args = parser.parse_args()

	#################################################################
	# Build SAM
	print("Building SAM...")
	model = build_sam(checkpoint=args.checkpoint_path)
	model.to(device="cuda")
	model = model
	predictor = SamPredictor(model)
	mask_generator = SamAutomaticMaskGenerator(model)
	print("Done building SAM.")

    #################################################################
	# Load the scene and establish link correspondences
	rgb_paths = natsorted(glob(os.path.join(args.rgb_folder, "*.jpg")))
	names_2d = list(map(get_sample_name, rgb_paths))
	if args.poses_folder:
		# Filter files based on valid camera poses
		poses_paths = get_paths(args.poses_folder, names_2d, ".txt")
		poses_paths = filter_poses(poses_paths)
		names_2d = list(map(get_sample_name, poses_paths))
		rgb_paths = get_paths(args.rgb_folder, names_2d, ".jpg")
	else:
		poses_paths = None
		


	#################################################################
	# Run SAM on the scene images

	# Prepare the canvas to draw the images
	img = cv2.imread(rgb_paths[0])
	height, width = img.shape[:2]

	fig = Figure(frameon=False)
	dpi = fig.get_dpi()
	scale = 1.0
	fig.set_size_inches(
		    (width * scale + 1e-2) / dpi,
		    (height * scale + 1e-2) / dpi,
		)
	canvas = FigureCanvas(fig)
	ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
	ax.axis("off")

	if not os.path.exists(args.output_folder):
		os.mkdir(args.output_folder)

	for path in rgb_paths:  # tqdm(rgb_paths):
		name = get_sample_name(path)
		img = cv2.imread(path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		start = time()
		masks = mask_generator.generate(img)
		print("SAM forward: {:.1f}s".format(time() - start))

		start = time()
		ax.imshow(img, extent=(0, width, height, 0), interpolation="nearest")
		for mask in masks:
			if True:
				# assign color randomly
				color = np.random.random(3)
			else:
				# assign color based on 3D overlap
				# with the instances assigned on previous steps
				pass

			add_mask_to_canvas(mask['segmentation'], ax, color)

		s, (width, height) = canvas.print_to_buffer()
		img_rgba = np.frombuffer(s, dtype="uint8")
		img_rgba = img_rgba.reshape(height, width, 4)
		rgb, alpha = np.split(img_rgba, [3], axis=2)
		rgb = rgb.astype("uint8")
		im2show = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

		out_name = "{:0>6}".format(name)
		output_path = os.path.join(args.output_folder, out_name + ".jpg")
		cv2.imwrite(output_path, im2show)
		print("Mask visualization: {:.1f}s".format(time() - start))
