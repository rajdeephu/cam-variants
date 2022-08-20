# imports
import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def grad_cam(img, model, conv_layer_name):
	# redefine model to get output of last conv layer
	cam_model = tf.keras.Model([model.inputs], [model.get_layer(conv_layer_name).output, model.output])

	# get gradients of predicted class
	with tf.GradientTape() as tape:
		conv_layer_output, preds = cam_model(img)
		pred_index = tf.argmax(preds[0])
		class_channel = preds[:, pred_index]
	grads = tape.gradient(class_channel, conv_layer_output)

	# grad cam
	pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
	conv_layer_output = conv_layer_output[0]
	heatmap = conv_layer_output @ pooled_grads[..., tf.newaxis]
	heatmap = tf.squeeze(heatmap)
	heatmap = tf.maximum(heatmap, 0)/tf.reduce_max(heatmap)
	return heatmap.numpy()

def layer_cam(img, model, conv_layer_name):
	# redefine model to get output of conv layer
	cam_model = tf.keras.Model([model.inputs], [model.get_layer(conv_layer_name).output, model.output])

	# get gradients of predicted class
	with tf.GradientTape() as tape:
		conv_layer_output, preds = cam_model(img)
		pred_index = tf.argmax(preds[0])
		class_channel = preds[:, pred_index]
	grads = tape.gradient(class_channel, conv_layer_output)

	# layer cam
	grads = tf.maximum(grads, 0)
	heatmap = tf.reduce_sum(tf.squeeze(grads) * conv_layer_output[0], axis=(2))
	heatmap = tf.maximum(heatmap, 0)/tf.reduce_max(heatmap)
	return heatmap.numpy()	

if __name__ == "__main__":

	# create the argument parser
	parser = argparse.ArgumentParser(description="CAM Variants")
	parser.add_argument("-g", "--gpu", choices=["0", "1", "2", "3"], help="gpu index", required=True)
	parser.add_argument("-t", "--type", choices=["gradcam", "layercam"], help="cam type", required=True)
	args = vars(parser.parse_args())

	# set gpu to use
	os.environ["CUDA_VISIBLE_DEVICES"] = args["gpu"]

	# use xception with imagenet weights
	model = tf.keras.applications.xception.Xception(weights="imagenet")
	preprocess_input = tf.keras.applications.xception.preprocess_input
	decode_predictions = tf.keras.applications.xception.decode_predictions
	model.layers[-1].activation = None
	input_size = (299, 299)

	# preprocess image
	img_path = tf.keras.utils.get_file(
		"sea_lion.jpg", 
		"https://nationalzoo.si.edu/sites/default/files/styles/1400x700_scale_and_crop/public/animals/californiasealion-001.jpg", 
		cache_dir=os.getcwd()
	)
	img = tf.keras.preprocessing.image.load_img(img_path, target_size=input_size)
	img = tf.keras.preprocessing.image.img_to_array(img)
	img = np.expand_dims(img, axis=0)
	img = preprocess_input(img)

	# get prediction of image
	preds = model.predict(img)
	print("Predicted:", decode_predictions(preds, top=1)[0])

	conv_layer_names = ["block13_sepconv1_act", "block13_sepconv2_act", "block14_sepconv1_act", "block14_sepconv2_act"]
	for layer in conv_layer_names:
		# get cam of image
		if args["type"] == "gradcam":
			heatmap = grad_cam(img, model, layer)
		elif args["type"] == "layercam":
			heatmap = layer_cam(img, model, layer)

		# plot heatmap
		# plt.matshow(heatmap)
		# plt.savefig(args["type"] + "/" + layer + ".jpg")

		# superimpose heatmap onto image
		original_image = tf.keras.preprocessing.image.load_img(img_path)
		original_image = tf.keras.preprocessing.image.img_to_array(original_image)
		heatmap = np.uint8(255 * heatmap)
		jet = cm.get_cmap("jet")
		jet_colors = jet(np.arange(256))[:, :3]
		jet_heatmap = jet_colors[heatmap]
		jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
		jet_heatmap = jet_heatmap.resize((original_image.shape[1], original_image.shape[0]))
		jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
		superimposed_img = jet_heatmap * 0.4 + original_image
		superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
		superimposed_img.save(args["type"] + "/" + layer + ".jpg")