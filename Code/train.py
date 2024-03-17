from keras.datasets import fashion_mnist
from keras.datasets import mnist
import wandb
import numpy as np
import argparse
import math
from FeedForwardNeuralNetwork import FFN
from PlotConfusionMatrix import plot_ConfusionMatrix


    

def train_model(args, config = None):
  """
  This function is used for trining the model with the provided wandb.config
  """

  if(args.dataset == "mnist"):
      (trainDataInput, trainDataOutput), (testDataInput, testDataOutput) = mnist.load_data()
      class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  else:
      (trainDataInput, trainDataOutput), (testDataInput, testDataOutput) = fashion_mnist.load_data()
      class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


  #Normalize the data
  trainDataInput = trainDataInput/np.max(trainDataInput)  
  testDataInput = testDataInput/np.max(testDataInput)
  #initialize a new wandb run
  with wandb.init(config = config, project=args.wandb_project, entity=args.wandb_entity):
    #config will be set by sweep controller if called by wandb.agent
    config = wandb.config
    run_name = "Test__opt_{}_eta_{}_actfunc_{}_init_{}_batch_{}_L2reg_{}_ep_{}_nHiddenL_{}_nNeurons_{}".format(config.optimizer, config.learning_rate, config.activation_func, config.weight_initialisation, config.batch_size, config.weight_decay, config.epochs, config.nHiddenLayers, config.nNeurons)
    wandb.run.name = run_name
    nSamples_90_percent = int(trainDataOutput.shape[0] * 0.9)
    trainX = trainDataInput[:nSamples_90_percent]
    trainY = trainDataOutput[:nSamples_90_percent]
    valX = trainDataInput[nSamples_90_percent:]
    valY = trainDataOutput[nSamples_90_percent:]
    model = FFN(trainX, trainY, config.nHiddenLayers, config.nNeurons, config.batch_size, config.weight_decay, config.epochs, config.learning_rate, config.activation_func, config.losstype, valX, valY, testDataInput, testDataOutput )
    model.initWeights(config.weight_initialisation)
    model.init_optimizer_params(args.momentum, args.beta, args.beta1, args.beta2, args.epsilon)
    model.train(config.optimizer)

    print(model.arrloss)
    for i in range(0, config.epochs):
        wandb.log({"Epoch":i, "Val_accuracy":model.arrvalAcc[i], "Val_Loss":model.arrvalLoss[i]})

    for i in range(0, config.epochs):
        wandb.log({"Epoch":i, "accuracy":model.arrtestAcc[i], "Loss":model.arrtestLoss[i]})

    print("Val_accuracy", model.arrvalAcc[-1])
    print("Test_accuracy", model.arrtestAcc[-1])
    wandb.log({"Val_accuracy":model.arrvalAcc[-1]})

    #Uncomment the below code to plot confusion matrix
    #(val_acc, val_loss) = model.test(valX,valY)
    # plot_ConfusionMatrix(model.testpred, valY, "Val_ConfusionMatrix")
    # img2 = plt.imread("Val_ConfusionMatrix.png")
    # wandb.log({"Val_ConfusionMatrix": wandb.Image(img2)})

    # (test_acc, test_loss) = model.test(testDataInput,testDataOutput)
    # plot_ConfusionMatrix(model.testpred, testDataOutput, "Test_ConfusionMatrix")
    # img2 = plt.imread("Test_ConfusionMatrix.png")
    # wandb.log({"Test_ConfusionMatrix": wandb.Image(img2)})

    wandb.run.save()
    wandb.run.finish()
  return

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train neural network with specified configurations')
    parser.add_argument('--wandb_project', '-wp', type=str, default='basic-intro', help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', '-we', type=str, default='drbruap', help='Weights & Biases entity')
    parser.add_argument('--dataset', '-d', type=str, default='fashion_mnist', choices=['mnist', 'fashion_mnist'], help='Dataset to use')
    parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size for training')
    parser.add_argument('--loss', '-l', type=str, default='cross_entropy', choices=['squared_error', 'cross_entropy'], help='Loss function')
    parser.add_argument('--optimizer', '-o', type=str, default='nadam', choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], help='Optimizer')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum for momentum and nag optimizers')
    parser.add_argument('--beta', '-beta', type=float, default=0.9, help='Beta for rmsprop optimizer')
    parser.add_argument('--beta1', '-beta1', type=float, default=0.9, help='Beta1 for adam and nadam optimizers')
    parser.add_argument('--beta2', '-beta2', type=float, default=0.999, help='Beta2 for adam and nadam optimizers')
    parser.add_argument('--epsilon', '-eps', type=float, default=0.0001, help='Epsilon for optimizers')
    parser.add_argument('--weight_decay', '-w_d', type=float, default=0.0, help='Weight decay for optimizers')
    parser.add_argument('--weight_init', '-w_i', type=str, default='xavier', help='Weight initialization method')
    parser.add_argument('--num_layers', '-nhl', type=int, default=3, help='Number of hidden layers')
    parser.add_argument('--hidden_size', '-sz', type=int, default=128, help='Number of hidden neurons in a layer')
    parser.add_argument('--activation', '-a', type=str, default='tanh', choices=['sigmoid', 'tanh', 'ReLU'], help='Activation function')

    args = parser.parse_args()

    config = {
    "epochs" :args.epochs,
    "nHiddenLayers" : args.num_layers,
    "nNeurons" :args.hidden_size,
    "weight_decay" : args.weight_decay,
    "learning_rate" : args.learning_rate,
    "optimizer" : args.optimizer,
    "batch_size" : args.batch_size,
    "weight_initialisation" : args.weight_init,
    "activation_func" : args.activation,
    "losstype" : args.loss
    }
    train_model(args, config)

  
def init_sweep():
    wandb.login(key = "017101a520090630fd58ad8684de73bf54c45117")
    sweep_config = { "name" : "FFN","method": "random"}
    metric = {
    "name" : "Val_accuracy",
    "goal" : "maximize"
    }
    sweep_config["metric"] = metric
    parameters_dict = {
    "epochs" : {"values" : [5,10]},
    "nHiddenLayers" : {"values":[3,4,5]},
    "nNeurons" :{"values" : [32,64,128]},
    "weight_decay" : {"values":[0, 0.0005, 0.5]},
    "learning_rate" : {"values":[1e-2, 1e-3, 1e-4]},
    "optimizer" : {"values":["sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"]},
    "batch_size" : {"values":[16, 32, 64]},
    "weight_initialisation" : {"values":["random", "xavier"]},
    "activation_func" : {"values" : ["sigmoid", "tanh", "ReLU"]}
    }
    sweep_config["parameters"] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="FeedForwardNeuralNetwork")
    wandb.agent(sweep_id, train_model, count=50)