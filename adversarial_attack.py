# gui python script to test untargeted and targeted attacks on image with ResNet50 model
# thanks for his tutorials https://www.pyimagesearch.com/

# import necessary packages
import cv2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.applications.resnet50 import decode_predictions
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf
import numpy as np
import argparse
import os
import json
import tkinter as Tk
from tkinter import ttk
from tkinter import filedialog as fd
from PIL import Image, ImageTk

image = None
imageOutput = None
classIdx = None

# load imagenet class index file
labelPath = os.path.join(os.path.dirname(__file__),"imagenet_class_index.json")
with open(labelPath) as f:
    imageNetClasses = {labels[1]: int(idx) for (idx, labels) in json.load(f).items()}

# convert image for processing
def preprocess_image(image):
	# swap color channels, resize the input image, and add a batch dimension
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))
	image = np.expand_dims(image, axis=0)
	return image

def clip_eps(tensor, eps):
	# clip the values of the tensor to a given range and return it
	return tf.clip_by_value(tensor, clip_value_min=-eps,
		clip_value_max=eps)

# update gui progress bar
def updateProgressBar(value):
	global root
	global progressbar

	progressbar["value"]=value
	style.configure('text.Horizontal.TProgressbar', text='{:g} %'.format(value))  # update label
	root.update_idletasks()

# function to generate untargeted adversarial attack
# for more explainations : https://www.pyimagesearch.com/2020/10/26/targeted-adversarial-attacks-with-keras-and-tensorflow/
def generate_untargeted_adversaries(model, baseImage, delta, classIdx, steps):
	# define the epsilon and learning rate constants
	EPS = 2 / 255.0
	LR = 0.1

	# initialize optimizer and loss function
	optimizer = Adam(learning_rate=LR)
	scc = SparseCategoricalCrossentropy()

	# iterate over the number of steps
	for step in range(0, steps):
		# record our gradients
		with tf.GradientTape() as tape:
			# explicitly indicate that our perturbation vector should
			# be tracked for gradient updates
			tape.watch(delta)

            # add our perturbation vector to the base image and
			# preprocess the resulting image
			adversary = preprocess_input(baseImage + delta)
			# run this newly constructed image tensor through our
			# model and calculate the loss with respect to the
			# *original* class index
			predictions = model(adversary, training=False)
			loss = -scc(tf.convert_to_tensor([classIdx]),
				predictions)
			# check to see if we are logging the loss value, and if
			# so, display it to our terminal
			if step % 5 == 0:
				print("step: {}, loss: {}...".format(step,
					loss.numpy()))
				updateProgressBar(100*step/steps)
		# calculate the gradients of loss with respect to the
		# perturbation vector
		gradients = tape.gradient(loss, delta)
		# update the weights, clip the perturbation vector, and
		# update its value
		optimizer.apply_gradients([(gradients, delta)])
		delta.assign_add(clip_eps(delta, eps=EPS))
	# return the perturbation vector
	return delta

# function to generate untargeted adversarial attack
# for more explainations : https://www.pyimagesearch.com/2020/10/19/adversarial-images-and-attacks-with-keras-and-tensorflow/
def generate_targeted_adversaries(model, baseImage, delta, classIdx, target, steps):
	# define the epsilon and learning rate constants
	EPS = 2 / 255.0
	LR = 0.005

	# initialize optimizer and loss function
	optimizer = Adam(learning_rate=LR)
	scc = SparseCategoricalCrossentropy()

	# iterate over the number of steps
	for step in range(0, steps):
		# record our gradients
		with tf.GradientTape() as tape:
			# explicitly indicate that our perturbation vector should
			# be tracked for gradient updates
			tape.watch(delta)

            # add our perturbation vector to the base image and
			# preprocess the resulting image
			adversary = preprocess_input(baseImage + delta)
			# run this newly constructed image tensor through our
			# model and calculate the loss with respect to the
			# *original* class index
			predictions = model(adversary, training=False)
			originalLoss = -scc(tf.convert_to_tensor([classIdx]),predictions)
			targetLoss = scc(tf.convert_to_tensor([target]),predictions)
			totalLoss = originalLoss + targetLoss

            # check to see if we are logging the loss value, and if
			# so, display it to our terminal
			if step % 5 == 0:
				print("step: {}, loss: {}...".format(step,
					totalLoss.numpy()))
				updateProgressBar(100*step/steps)
		# calculate the gradients of loss with respect to the
		# perturbation vector
		gradients = tape.gradient(totalLoss, delta)
		# update the weights, clip the perturbation vector, and
		# update its value
		optimizer.apply_gradients([(gradients, delta)])
		delta.assign_add(clip_eps(delta, eps=EPS))
	# return the perturbation vector
	return delta

# create tkinter window
root = Tk.Tk()
root.title("Untargeted adversarial attack")
root.geometry("800x600")

# load the input image from disk and preprocess it
def loadImage(filename):
	global image
	print("loading image...")
	imageFile = cv2.imread(filename)
	image = preprocess_image(imageFile)
	img = Image.open(filename)
	img.thumbnail((400,500))
	img = ImageTk.PhotoImage(img)
	canvasInput.create_image(200,250, anchor=Tk.CENTER, image=img)
	canvasInput.img = img

# classify image with the ResNet50 model
def classify():
	global image
	global labelInput
	global classIdx
	global root
	global progressbar
	global model

	print("classifing image...")

	progressbar.pack(side=Tk.LEFT,padx=5,pady=5,expand=True)

	updateProgressBar(25)

	# load the pre-trained ResNet50 model for running inference
	print("loading pre-trained ResNet50 model")
	model = ResNet50(weights="imagenet")

	updateProgressBar(50)

	preds = model.predict(image)
	P = decode_predictions(preds)
	(imagenetID, label, prob) = P[0][0]
	classIdx = imageNetClasses[label]
	labelInput.config(text="{:} : {:.2f}%".format(label,100*prob))

	progressbar.pack_forget()

# process attack with untargeted or target function
def attack():
	global combobox
	global classIdx
	global imageOutput
	global progressbar
	global root

	progressbar.pack(side=Tk.LEFT,padx=5,pady=5,expand=True)

	# create a tensor based off the input image and initialize the
	# perturbation vector (we will update this vector via training)
	baseImage = tf.constant(image, dtype=tf.float32)
	delta = tf.Variable(tf.zeros_like(baseImage), trainable=True)
	# generate the perturbation vector to create an adversarial example
	print("generating perturbation...")
	if combobox.get()=="Random":
		deltaUpdated = generate_untargeted_adversaries(model, baseImage, delta, classIdx, 50)
	else:
		deltaUpdated = generate_targeted_adversaries(model, baseImage, delta, classIdx, imageNetClasses[combobox.get()], 100)

	# create the adversarial example, swap color channels, and save the
	# output image to disk
	print("creating adversarial image")
	adverImage = (baseImage + deltaUpdated).numpy().squeeze()
	adverImage = np.clip(adverImage, 0, 255).astype("uint8")
	imageOutput = adverImage

	# display created image on gui
	img = ImageTk.PhotoImage(image=Image.fromarray(np.array(imageOutput)))
	canvasOutput.create_image(200,250, anchor=Tk.CENTER, image=img)
	canvasOutput.img = img

	# run inference with this adversarial example, parse the results,
	# and display the top-1 predicted result
	print("running inference on the adversarial image")
	preprocessedImage = preprocess_input(baseImage + deltaUpdated)
	predictions = model.predict(preprocessedImage)
	predictions = decode_predictions(predictions, top=3)[0]
	label = predictions[0][1]
	confidence = predictions[0][2] * 100
	labelOutput.config(text="{:} : {:.2f}%".format(label,confidence))
	progressbar.pack_forget()

# open window select file and load it
def openAndDisplay():
	files = [('All', '*.*'),
				('Png', '*.png'),
				('Jpg', '*.jpg'),
				('Jpeg', '*.jpeg')]
	filename = fd.askopenfilename(filetypes = files, defaultextension = files)

	loadImage(filename)

# save image to file
def save():
	global imageOutput
	files = [('Png', '*.png'),
				('Jpg', '*.jpg')]
	filename = fd.asksaveasfilename(filetypes = files, defaultextension = files)

	image = cv2.cvtColor(imageOutput, cv2.COLOR_RGB2BGR)

	cv2.imwrite(filename, image)

# handle combobox change to update title
def comboboxSelected(event):
	global root
	global combobox

	if combobox.get()=="Random":
		root.title("Untargeted adversarial attack")
	else:
		root.title("Targeted adversarial attack")

# gui widgets initialization
frameButtons = Tk.Frame(height=50)
frameButtons.pack(fill=Tk.X,padx=5,pady=5)

fileOpen = Tk.Button(frameButtons, text='Click to open File',command=openAndDisplay)
fileOpen.pack(side=Tk.LEFT,padx=5,pady=5)

detect = Tk.Button(frameButtons, text='Classify',command=classify)
detect.pack(side=Tk.LEFT,padx=5,pady=5)

combobox = ttk.Combobox(frameButtons)
combobox.pack(side=Tk.LEFT,padx=5,pady=5)
combobox["values"] = tuple(["Random"] + list(imageNetClasses.keys()))
combobox.current(0)
combobox.bind("<<ComboboxSelected>>", comboboxSelected)

attack = Tk.Button(frameButtons, text='Attack',command=attack)
attack.pack(side=Tk.LEFT,padx=5,pady=5)

save = Tk.Button(frameButtons, text='Save',command=save)
save.pack(side=Tk.LEFT,padx=5,pady=5)

style = ttk.Style(root)

# add label in the layout
style.layout('text.Horizontal.TProgressbar', 
             [('Horizontal.Progressbar.trough',
               {'children': [('Horizontal.Progressbar.pbar',
                              {'side': 'left', 'sticky': 'ns'})],
                'sticky': 'nswe'}), 
              ('Horizontal.Progressbar.label', {'sticky': ''})])

# set initial text
style.configure('text.Horizontal.TProgressbar', text='0 %')
progressbar=ttk.Progressbar(frameButtons,cursor="watch",orient="horizontal",style='text.Horizontal.TProgressbar')
progressbar["value"]=0
progressbar["maximum"]=100
progressbar.pack(side=Tk.LEFT,padx=5,pady=5,expand=True)
progressbar.pack_forget()

frameCanvas = Tk.Frame(height=500)
frameCanvas.pack(fill=Tk.X,padx=0,pady=0)

canvasInput = Tk.Canvas(frameCanvas, width = 400, height = 500) 
canvasInput.pack(side=Tk.LEFT,padx=0,pady=0)

canvasOutput = Tk.Canvas(frameCanvas, width = 400, height = 500) 
canvasOutput.pack(side=Tk.RIGHT,padx=0,pady=0)

frameResults = Tk.Frame()
frameResults.pack(fill=Tk.BOTH,padx=0,pady=0)

labelInput = Tk.Label(frameResults, text="")
labelInput.pack(side=Tk.LEFT,padx=5,pady=5,expand=True)

labelOutput = Tk.Label(frameResults, text="")
labelOutput.pack(side=Tk.RIGHT,padx=5,pady=5,expand=True)

root.mainloop()
