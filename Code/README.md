This Github repository presents the codes for assignment 1 of CS6910.

### train.py
This python file can be executed to train a FFN model by passing required arguments as mentioned below

| Name                | Description                                                                           |
|---------------------|---------------------------------------------------------------------------------------|
| `wp`, `wandb_project` | Project name used to track experiments in Weights & Biases dashboard                |
| `we`, `wandb_entity`  | Wandb Entity used to track experiments in the Weights & Biases dashboard             |
| `d`, `dataset`         | Dataset used for training (`mnist` or `fashion_mnist`)                               |
| `e`, `epochs`          | Number of epochs to train the neural network                                         |
| `b`, `batch_size`      | Batch size used for training the neural network                                      |
| `l`, `loss`            | Loss function used (`mean_squared_error` or `cross_entropy`)                          |
| `o`, `optimizer`       | Optimizer used for training (`sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam`)    |
| `lr`, `learning_rate`  | Learning rate used for optimizing model parameters                                    |
| `m`, `momentum`        | Momentum used by momentum and nag optimizers                                           |
| `beta`                 | Beta used by rmsprop optimizer                                                        |
| `beta1`, `beta2`       | Beta values used by adam and nadam optimizers                                          |
| `eps`, `epsilon`       | Epsilon used by optimizers                                                            |
| `wd`, `weight_decay`   | Weight decay used by optimizers                                                        |
| `wi`, `weight_init`    | Weight initialization method (`random` or `Xavier`)                                    |
| `nhl`, `num_layers`    | Number of hidden layers used in the feedforward neural network                         |
| `sz`, `hidden_size`    | Number of hidden neurons in a feedforward layer                                        |
| `a`, `activation`      | Activation function used (`identity`, `sigmoid`, `tanh`, `ReLU`)                       |



-------------------------------------------------
### FeedForwardNeuralNetwrok.py 
contains class FFN which represents the neural network model. 
train method is used to train the model with optimizer as input
test method can be used to test the model with input x and y as parameters

-----------------------------------------------------------------------
### PlotConfusionMatrix.py 
contains method to plot the confusion matrix provided true and predicted

-----
### SampleImagePlot.py
plots sample image of each class type in the given dataset
