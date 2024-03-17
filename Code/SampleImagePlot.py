import wandb
import numpy as np
from keras.datasets import fashion_mnist
wandb.login(key = "017101a520090630fd58ad8684de73bf54c45117")

(trainDataInput, trainDataOutput), (testDataInput, testDataOutput) = fashion_mnist.load_data()

#Normalize the data
trainDataInput = trainDataInput/np.max(trainDataInput)
testDataInput = testDataInput/np.max(testDataInput)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

wandb.init(project="FeedForwardNeuralNetwork",id="PlotSampleImages")

def log_images():
  images=[]
  labels=[]
  count=0
  num_classes = len(class_names)
  for i in range(num_classes):
    #Get the list of indices where label i is present
    class_indices = np.where(trainDataOutput == i)[0]
    if class_indices.size == 0:
            continue  # Skip if there are no samples for this class
    #Use the first index and plot the respective image
    index = class_indices[0]
    images.append(trainDataInput[index])
    labels.append(class_names[i])
  wandb.log({"Plot": [wandb.Image(img, caption=caption) for img, caption in zip(images, labels)]})
  return

log_images()
