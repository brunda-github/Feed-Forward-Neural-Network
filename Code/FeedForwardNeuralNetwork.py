import numpy as np


class FFN:
  def __init__(self, input, output, nHiddenLayers, nNeurons, batchsize, reg_lambda = 0, nEpochs = 50,eta = 0.001,activationfunc = "sigmoid"):
    self.x = input
    self.y = output
    self.nLayers = nHiddenLayers + 1 # +1 for output layer

    if isinstance(nNeurons, int):
      #same number of neurons in all hidden layers
      self.nNeurons = np.full(nHiddenLayers, nNeurons)
      self.nNeurons = np.insert(self.nNeurons,0,input[0].flatten().shape[0])
      self.nNeurons = np.append(self.nNeurons, 10)
    elif len(nNeurons) == nHiddenLayers:
      #different numer of neurons in each hidden layer
      self.nNeurons = np.insert(nNeurons,0,input[0].flatten().shape[0])
      self.nNeurons = np.append(self.nNeurons, 10)
    else:
      raise IndexError("Incorrect layers and neurons per layer")

    self.activationfunc = activationfunc
    self.batchsize = batchsize
    self.eta = eta
    self.nEpochs = nEpochs
    self.reg_lambda = reg_lambda
    self.a = np.empty(self.nLayers+1, dtype = object)
    self.h = np.empty(self.nLayers+1, dtype = object)
    self.b = np.empty(self.nLayers+1, dtype = object)
    self.w = np.empty(self.nLayers+1, dtype = object)
    self.h[0] = input[0].flatten()
    self.arrloss = []
    return

  def initWeights(self, type):
    if type == "random":
      for k in range(1, self.nLayers + 1):
        self.w[k] = np.random.randn(self.nNeurons[k], self.nNeurons[k-1])
        self.b[k] = np.random.rand(self.nNeurons[k])
        # self.w[k] = np.random.uniform(low = -1, high = 1, size = (self.nNeurons[k], self.nNeurons[k-1]))
        # self.b[k] = np.random.uniform(low = -1, high = 1, size = self.nNeurons[k])
    if type == "ones":
      for k in range(1, self.nLayers + 1):
        self.w[k] = np.ones((self.nNeurons[k], self.nNeurons[k-1]))
        self.b[k] = np.ones(self.nNeurons[k])
    if type == "xavier":
      for k in range(1, self.nLayers + 1):
        std_dev = np.sqrt(2.0/(self.nNeurons[k] + self.nNeurons[k-1]))
        self.w[k] = np.random.normal(loc = 0.0, scale = std_dev, size = (self.nNeurons[k], self.nNeurons[k-1]))
        self.b[k] = np.zeros(self.nNeurons[k])
    return

  def sigmoid(x):
    return [1 / (1 + np.exp(-x[i])) for i in range(len(x))]

  def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=0)

  def activationfunc_g(self,x):
    if self.activationfunc == "sigmoid":
      return 1 / (1 + np.exp(-x))
    elif self.activationfunc == "tanh":
      #return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
      return np.tanh(x)
    elif self.activationfunc == "ReLU":
      return np.maximum(0,x)
    return

  def gprime(self,x):
    if self.activationfunc == "sigmoid":
      sigmoid_x = 1 / (1 + np.exp(-x))
      return sigmoid_x * (1 - sigmoid_x)
    elif self.activationfunc == "tanh":
      return 1 - np.tanh(x)**2
    elif self.activationfunc == "ReLU":
      return np.where(x <= 0, 0, 1)
    return

  def Outputfunc(self, x):
    #Softmax
    #Subtract the maximum value from each element of x to prevent large exponentials that can lead to overflow.
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=0)


  def forward_propogation(self, index):
    l = self.nLayers
    self.h[0] = self.x[index].flatten()

    for k in range(1, l):
      self.a[k] = np.dot(self.w[k], self.h[k-1]) + self.b[k]
      self.h[k] = self.activationfunc_g(self.a[k])
    self.a[l] = np.dot(self.w[l], self.h[l-1]) + self.b[l]

    self.ypred = self.Outputfunc(self.a[l])
    one_hot_vec = np.zeros(self.nNeurons[l], dtype = float)
    one_hot_vec[self.y[index]] = 1
    loss = -sum([one_hot_vec[i]*np.log2(self.ypred[i]+ 1e-9) for i in range(len(one_hot_vec))])
    return loss

  def back_propogation(self, index):
    delw = np.empty(self.w.shape, dtype = object)
    delb = np.empty(self.b.shape, dtype = object)
    l = self.nLayers

    #Compute output gradient
    one_hot_vec = np.zeros(self.nNeurons[l], dtype = float)
    one_hot_vec[self.y[index]] = 1
    dela = -(one_hot_vec - self.ypred)
    delh = dela
    for k in range(l,0,-1):
      #Compute gradients w.r.t parameters
      delw[k] = np.dot(dela.reshape(-1,1), self.h[k-1].reshape(1,-1)) + self.reg_lambda*self.w[k]
      delb[k] = dela
      #Compute gradients w.r.t layer below
      if k!=1:
        delh = np.dot(np.transpose(self.w[k]),dela)
        dela = delh * self.gprime(self.a[k-1])
    return (delw, delb)

  def do_SGD(self):
    t = 0
    eta = self.eta
    nIterations = self.nEpochs
    while t < nIterations:
      gradw = np.zeros(self.w.shape, dtype = object)
      gradb = np.zeros(self.b.shape, dtype = object)
      totalLoss = 0
      for i in range(self.x.shape[0]):
        loss = self.forward_propogation(i)
        (delw, delb) = self.back_propogation(i)
        totalLoss += loss
        for k in range(1, self.nLayers + 1):
          gradw[k] += delw[k]
          gradb[k] += delb[k]
          if (i+1)%self.batchsize == 0:
            self.w[k] = self.w[k] - eta * gradw[k]
            self.b[k] = self.b[k] - eta * gradb[k]
            gradw = np.zeros(self.w.shape, dtype = object)
            gradb = np.zeros(self.b.shape, dtype = object)
      if self.x.shape[0] %self.batchsize != 0:
        for k in range(1, self.nLayers + 1):
          self.w[k] = self.w[k] - eta * gradw[k]
          self.b[k] = self.b[k] - eta * gradb[k]

      t += 1
      self.arrloss.append(totalLoss/self.x.shape[0])
      print("Epoch",t, "loss",totalLoss/self.x.shape[0])
    return

  def do_momentum(self):
    t = 0
    prev_updw = np.zeros(self.w.shape, dtype = object)
    prev_updb = np.zeros(self.b.shape, dtype = object)
    updw =  np.zeros(self.w.shape, dtype = object)
    updb = np.zeros(self.b.shape, dtype = object)
    beta = 0.9
    eta = self.eta
    nIterations = self.nEpochs
    while t < nIterations:
      gradw = np.zeros(self.w.shape, dtype = object)
      gradb = np.zeros(self.b.shape, dtype = object)
      totalLoss = 0
      for i in range(self.x.shape[0]):
          loss = self.forward_propogation(i)
          (delw, delb) = self.back_propogation(i)
          totalLoss += loss

          for k in range(1, self.nLayers + 1):
            gradw[k] += delw[k]
            gradb[k] += delb[k]
            if (i+1)%self.batchsize == 0:
              updw[k] = beta * prev_updw[k] + eta * gradw[k]
              updb[k] = beta * prev_updb[k] + eta * gradb[k]
              self.w[k] = self.w[k] -  updw[k]
              self.b[k] = self.b[k] -  updb[k]
              prev_updw[k] = updw[k]
              prev_updb[k] = updb[k]
              gradw = np.zeros(self.w.shape, dtype = object)
              gradb = np.zeros(self.b.shape, dtype = object)

      if self.x.shape[0] %self.batchsize != 0:
        for k in range(1, self.nLayers + 1):
          updw[k] = beta * prev_updw[k] + eta * gradw[k]
          updb[k] = beta * prev_updb[k] + eta * gradb[k]
          self.w[k] = self.w[k] -  updw[k]
          self.b[k] = self.b[k] -  updb[k]
          prev_updw[k] = updw[k]
          prev_updb[k] = updb[k]

      t+=1
      self.arrloss.append(totalLoss/self.x.shape[0])
      print("Epoch",t, "loss",totalLoss/self.x.shape[0])
    return

  def do_NAG(self):
    t = 0
    prev_updw = np.zeros(self.w.shape, dtype = object)
    prev_updb = np.zeros(self.b.shape, dtype = object)
    updw =  np.zeros(self.w.shape, dtype = object)
    updb = np.zeros(self.b.shape, dtype = object)
    beta = 0.9
    nIterations = self.nEpochs
    eta = self.eta
    while t < nIterations:
      gradw = np.zeros(self.w.shape, dtype = object)
      gradb = np.zeros(self.b.shape, dtype = object)
      totalLoss = 0


      #Iterate through the data and find the gradient
      for i in range(self.x.shape[0]):
        if (i)%self.batchsize == 0 :
          #Update the prev history first
          for k in range(1, self.nLayers + 1):
            self.w[k] = self.w[k] - beta * prev_updw[k]
            self.b[k] = self.b[k] - beta * prev_updb[k]
        loss = self.forward_propogation(i)
        (delw, delb) = self.back_propogation(i)
        totalLoss += loss
        for k in range(1, self.nLayers + 1):
          gradw[k] += delw[k]
          gradb[k] += delb[k]
          if (i+1)%self.batchsize == 0 :
            #History is already updated so update only w.r.t gradient
            self.w[k] = self.w[k] - eta * gradw[k]
            self.b[k] = self.b[k] - eta * gradb[k]
            updw[k] = beta * prev_updw[k] + eta * gradw[k]
            updb[k] = beta * prev_updb[k] + eta * gradb[k]
            prev_updw[k] = updw[k]
            prev_updb[k] = updb[k]

            gradw = np.zeros(self.w.shape, dtype = object)
            gradb = np.zeros(self.b.shape, dtype = object)

      if self.x.shape[0] %self.batchsize != 0:
        for k in range(1, self.nLayers + 1):
          #History is already updated so update only w.r.t gradient
          self.w[k] = self.w[k] - eta * gradw[k]
          self.b[k] = self.b[k] - eta * gradb[k]
          updw[k] = beta * prev_updw[k] + eta * gradw[k]
          updb[k] = beta * prev_updb[k] + eta * gradb[k]
          prev_updw[k] = updw[k]
          prev_updb[k] = updb[k]

      self.arrloss.append(totalLoss/self.x.shape[0])
      print("Epoch",t, "loss",totalLoss/self.x.shape[0])
      t += 1
    return

  def do_RMSProp(self):
    v_w = np.zeros(self.w.shape, dtype = object)
    v_b = np.zeros(self.b.shape, dtype = object)
    beta = 0.9
    eps = 1e-4
    t = 0
    eta = self.eta
    nIterations = self.nEpochs
    while t < nIterations:
      gradw = np.zeros(self.w.shape, dtype = object)
      gradb = np.zeros(self.b.shape, dtype = object)
      totalLoss = 0
      for i in range(self.x.shape[0]):
        loss = self.forward_propogation(i)
        (delw, delb) = self.back_propogation(i)
        totalLoss += loss
        for k in range(1, self.nLayers + 1):
          gradw[k] += delw[k]
          gradb[k] += delb[k]
          if (i+1)%self.batchsize == 0:
            v_w[k] = beta * v_w[k] + (1-beta)*np.square(gradw[k])
            v_b[k] = beta * v_b[k] + (1-beta)*np.square(gradb[k])
            self.w[k] = self.w[k] - eta * (1/np.sqrt(v_w[k] + eps)) * gradw[k]
            self.b[k] = self.b[k] - eta *(1/np.sqrt(v_b[k] + eps))* gradb[k]

            gradw = np.zeros(self.w.shape, dtype = object)
            gradb = np.zeros(self.b.shape, dtype = object)
      if(self.x.shape[0]%self.batchsize != 0):
        for k in range(1, self.nLayers+1):
          v_w[k] = beta * v_w[k] + (1-beta)*np.square(gradw[k])
          v_b[k] = beta * v_b[k] + (1-beta)*np.square(gradb[k])
          self.w[k] = self.w[k] - eta * (1/np.sqrt(v_w[k] + eps)) * gradw[k]
          self.b[k] = self.b[k] - eta *(1/np.sqrt(v_b[k] + eps))* gradb[k]

      self.arrloss.append(totalLoss/self.x.shape[0])
      print("Epoch",t, "loss",totalLoss/self.x.shape[0])
      t += 1
    return

  def do_adam(self):
    prev_updw = np.zeros(self.w.shape, dtype = object)
    prev_updb = np.zeros(self.b.shape, dtype = object)
    v_w = np.zeros(self.w.shape, dtype = object)
    v_b = np.zeros(self.b.shape, dtype = object)
    m_w = np.zeros(self.w.shape, dtype = object)
    m_b = np.zeros(self.b.shape, dtype = object)
    v_w_hat = np.zeros(self.w.shape, dtype = object)
    v_b_hat = np.zeros(self.b.shape, dtype = object)
    m_w_hat = np.zeros(self.w.shape, dtype = object)
    m_b_hat = np.zeros(self.b.shape, dtype = object)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-4
    t = 0
    step = 1
    iter_mixed = 5;
    ndata = 0
    eta = self.eta
    nIterations = self.nEpochs
    while t < nIterations:
      gradw = np.zeros(self.w.shape, dtype = object)
      gradb = np.zeros(self.b.shape, dtype = object)
      totalLoss = 0

      for i in range(self.x.shape[0]):
        loss = self.forward_propogation(i)
        (delw, delb) = self.back_propogation(i)
        totalLoss += loss
        for k in range(1, self.nLayers + 1):
          gradw[k] += delw[k]
          gradb[k] += delb[k]
          if (i+1)%self.batchsize == 0:
            v_w[k] = beta2 * v_w[k] + (1-beta2)*np.square(gradw[k])
            v_b[k] = beta2 * v_b[k] + (1-beta2)*np.square(gradb[k])
            m_w[k] = beta1 * m_w[k] + (1-beta1)*gradw[k]
            m_b[k] = beta1 * m_b[k] + (1-beta1)*gradb[k]

            v_w_hat[k] = v_w[k]/(1-np.power(beta2, step))
            v_b_hat[k] = v_b[k]/(1-np.power(beta2, step))
            m_w_hat[k] = m_w[k]/(1-np.power(beta1, step))
            m_b_hat[k] = m_b[k]/(1-np.power(beta1, step))

            self.w[k] = self.w[k] - eta * (1/np.sqrt(v_w_hat[k] + eps)) * m_w_hat[k]
            self.b[k] = self.b[k] - eta *(1/np.sqrt(v_b_hat[k] + eps))* m_b_hat[k]

            gradw = np.zeros(self.w.shape, dtype = object)
            gradb = np.zeros(self.b.shape, dtype = object)
            step += 1

      if self.x.shape[0]%self.batchsize != 0:
        for k in range(1, self.nLayers+1):
          v_w[k] = beta2 * v_w[k] + (1-beta2)*np.square(gradw[k])
          v_b[k] = beta2 * v_b[k] + (1-beta2)*np.square(gradb[k])
          m_w[k] = beta1 * m_w[k] + (1-beta1)*delw[k]
          m_b[k] = beta1 * m_b[k] + (1-beta1)*delb[k]

          v_w_hat[k] = v_w[k]/(1-np.power(beta2, t+1))
          v_b_hat[k] = v_b[k]/(1-np.power(beta2, t+1))
          m_w_hat[k] = m_w[k]/(1-np.power(beta1, t+1))
          m_b_hat[k] = m_b[k]/(1-np.power(beta1, t+1))

          self.w[k] = self.w[k] - eta * (1/np.sqrt(v_w_hat[k] + eps)) * m_w_hat[k]
          self.b[k] = self.b[k] - eta *(1/np.sqrt(v_b_hat[k] + eps))* m_b_hat[k]

      self.arrloss.append(totalLoss/self.x.shape[0])
      print("Epoch",t, "loss",totalLoss/self.x.shape[0])
      t+=1
    return

  def do_nadam(self):
    prev_updw = np.zeros(self.w.shape, dtype = object)
    prev_updb = np.zeros(self.b.shape, dtype = object)
    v_w = np.zeros(self.w.shape, dtype = object)
    v_b = np.zeros(self.b.shape, dtype = object)
    m_w = np.zeros(self.w.shape, dtype = object)
    m_b = np.zeros(self.b.shape, dtype = object)
    v_w_hat = np.zeros(self.w.shape, dtype = object)
    v_b_hat = np.zeros(self.b.shape, dtype = object)
    m_w_hat = np.zeros(self.w.shape, dtype = object)
    m_b_hat = np.zeros(self.b.shape, dtype = object)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-10
    t = 0
    iter_mixed = 5
    ndata = 0
    eta = self.eta
    step = 1
    nIterations = self.nEpochs
    while t < nIterations:
      gradw = np.zeros(self.w.shape, dtype = object)
      gradb = np.zeros(self.b.shape, dtype = object)
      totalLoss = 0

      for i in range(self.x.shape[0]):
        loss = self.forward_propogation(i)
        (delw, delb) = self.back_propogation(i)
        totalLoss += loss
        for k in range(1, self.nLayers + 1):
          gradw[k] += delw[k]
          gradb[k] += delb[k]
          if (i+1)%self.batchsize == 0:
            v_w[k] = beta2 * v_w[k] + (1-beta2)*np.square(gradw[k])
            v_b[k] = beta2 * v_b[k] + (1-beta2)*np.square(gradb[k])
            m_w[k] = beta1 * m_w[k] + (1-beta1)*gradw[k]
            m_b[k] = beta1 * m_b[k] + (1-beta1)*gradb[k]

            v_w_hat[k] = v_w[k]/(1-np.power(beta2, step))
            v_b_hat[k] = v_b[k]/(1-np.power(beta2, step))
            m_w_hat[k] = m_w[k]/(1-np.power(beta1, step))
            m_b_hat[k] = m_b[k]/(1-np.power(beta1, step))

            self.w[k] = self.w[k] - eta * (1/np.sqrt(v_w_hat[k] + eps)) *(beta1*m_w_hat[k] + (1-beta1)*gradw[k]/(1-np.power(beta1, step)))
            self.b[k] = self.b[k] - eta *(1/np.sqrt(v_b_hat[k] + eps))* (beta1*m_b_hat[k]+ (1-beta1)*gradb[k]/(1-np.power(beta1, step)))

            gradw = np.zeros(self.w.shape, dtype = object)
            gradb = np.zeros(self.b.shape, dtype = object)
            step +=1

      if self.x.shape[0]%self.batchsize != 0:
        for k in range(1, self.nLayers+1):
          v_w[k] = beta2 * v_w[k] + (1-beta2)*np.square(gradw[k])
          v_b[k] = beta2 * v_b[k] + (1-beta2)*np.square(gradb[k])
          m_w[k] = beta1 * m_w[k] + (1-beta1)*delw[k]
          m_b[k] = beta1 * m_b[k] + (1-beta1)*delb[k]

          v_w_hat[k] = v_w[k]/(1-np.power(beta2, t+1))
          v_b_hat[k] = v_b[k]/(1-np.power(beta2, t+1))
          m_w_hat[k] = m_w[k]/(1-np.power(beta1, t+1))
          m_b_hat[k] = m_b[k]/(1-np.power(beta1, t+1))

          self.w[k] = self.w[k] - eta * (1/np.sqrt(v_w_hat[k] + eps)) *(beta1*m_w_hat[k] + (1-beta1)*gradw[k]/(1-np.power(beta1, step)))
          self.b[k] = self.b[k] - eta *(1/np.sqrt(v_b_hat[k] + eps))* (beta1*m_b_hat[k]+ (1-beta1)*gradb[k]/(1-np.power(beta1, step)))


      self.arrloss.append(totalLoss/self.x.shape[0])
      print("Epoch",t, "loss",totalLoss/self.x.shape[0])
      t+=1
    return

  def train(self, alg = "sgd"):
    if(alg == "sgd"):
      self.do_SGD()
    elif(alg == "momentum"):
      self.do_momentum()
    elif(alg=="nesterov"):
      self.do_NAG()
    elif(alg == "rmsprop"):
      self.do_RMSProp()
    elif(alg == "adam"):
      self.do_adam()
    elif(alg == "nadam"):
      self.do_nadam()
    return

  def test(self, x, y):
    crctpred = 0;
    totalloss = 0;
    for i in range(0,y.shape[0]):
      l = self.nLayers
      self.h[0] = x[i].flatten()
      #print(self.h[0])
      for k in range(1, l):
        self.a[k] = np.dot(self.w[k], self.h[k-1]) + self.b[k]
        self.h[k] = self.activationfunc_g(self.a[k])
      self.a[l] = np.dot(self.w[l], self.h[l-1]) + self.b[l]
      self.ypred = self.Outputfunc(self.a[l])
      one_hot_vec = np.zeros(self.nNeurons[l], dtype = float)
      one_hot_vec[self.y[i]] = 1
      loss = -sum([one_hot_vec[k]*np.log2(self.ypred[k]+ 1e-9) for k in range(len(one_hot_vec))])
      totalloss += loss
      # print("prediction: {}", np.argmax(self.ypred))
      # print("True Label: {}", y[i])
      if(np.argmax(self.ypred) == y[i]):
        crctpred+=1

    accuracy = crctpred/y.shape[0] * 100
    totalloss /= y.shape[0]
    return (accuracy, totalloss)

