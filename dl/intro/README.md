 ## What is neural network?
 X, y, w, b -----> Z = w.T * X + b ------> A = sigmoid(Z) -----> loss(A, y) 
 
 file logisticregressionNNminset.py is an example that shows a simple mindset of neural network implementation 
 
 
 single hiden layer NN
<img src="https://user-images.githubusercontent.com/64529936/119261066-f6ebaa80-bbd5-11eb-962f-0641b5a61f57.png" width="300" height="300">

input layer input featurs to hiden layers

first node in hiden layers computes like z1 = w1.T * x1 + b1,  a1 = sigmoid(z1)  and second node and the rest computed in the same way
(Z is a column vector, w is a (nxn) matric, b is a (nx1) array )
the horizontally the matrix A goes over different training examples and vertically the different indices in the matrix A corresponds to different hidden units.
  
# Forward propagation:  
  
z[1]= w[1]x +b[1] ---
a[1] =sigmid[z[1]] ---
z[2] w[2]a[1]+ b[2] ---
a[2] = simoid[z[2]] 

# Activation functions

<img src= "https://user-images.githubusercontent.com/64529936/119306768-f2bd9c80-bc6a-11eb-9ff4-1495b5da7061.png" width="500" height="500">

a = sigmoid(z) =. 1/1+exp(-z).      ------ binary calssificaton y = {0,1}

a = tanh(z)  = exp(z)-exp(-z)/exp(z)+exp(-z) -------- -1<y<1

                                
a = max{0,z}.  --->relu ------> computation is faster , learning nn is faster becous of slop of relu or leaky relu.

# Backward propagation:

dsigmoid/dz = give the slop of function ot z. ==. 1/1+exp(-z)(1-1/1+exp(-z)) = a(z)(1-a(z))

if z is large so a(z) = 1 and  da/dz = 0  and if z is near to zero then a(z) = 1/2  and da/dz = 1/4. 

a is activation function and a' is dirivative ----> 1-a

in tanh activation

 da/dz = 1-(tanh(z))^2.   
 
 if z is large a(z) = 1 and da/dz = 1- (1)^2 = 0 
 and if z is near to zero a(z) = 0 and da/dz = 1
 so a is activation and a' = 1- a^2
 
 relu is activation function
 
 a = max(0,z)
 if z < 0  a' = 0 and if z>= 0 a' = 1
 
 # Gradient descent nn
 parameters : w[1],b[1], w[2],b[2].       n[0], hiden layer n[1],  output layer n[1] = 1
 
 dimension of parameter w[1]--->(n[1],n[0]), b[1]--->(n[0],1),  w[2]--->(n[2],n[1]), b[2]--->(n[1],1)
 
 cost function : J(w[2],b[2],w[1],b[1]) = 1/m sum(L(yhat-y))
 
 gredient desent : repeat{ compute predect ( yhat(i) --- i=0 .... m )
 
                                            dw[1] = dJ/dw[1], db[1] =dJ/db[1], dw[2]= dJ/dw[2] , db = dJ/db[2]
                                            
                                            w[1] = w[1] - alpha dw[1] , b[1] = b[1] - alpha db[1]
  
  
 
da = dLoss(a,y)/da= -yloga - (1-y)log(1-a)


The general methodology to build a Neural Network is to:

1. Define the neural network structure ( # of input units,  # of hidden units, etc). 

2. Initialize the model's parameters

3. Loop:
    - Implement forward propagation
    - Compute loss
    - Implement backward propagation to get the gradients
    - Update parameters (gradient descent)
 
 ![image](https://user-images.githubusercontent.com/64529936/119332172-8782c300-bc88-11eb-8e2a-3ab7ebf7f902.png)
in tanh(z) : To compute dZ1 we need to compute g[1]'(????[1]) . Since g[1](.)  is the tanh activation function  then g[1]'(????)=1???????^2 . So we can compute g[1]'(????[1])  using (1 - np.power(A1, 2)).

## Classic network
1- LeNet-5 , 2- AlexNet, 3- VGG

ResNet :
The core idea of ResNet is introducing a so-called ???identity shortcut connection??? that skips one or more layers.
Very, very deep neural networks are difficult to train because of vanishing and exploding gradient types of problems.
Using residual blocks allows you to train much deeper neural networks.

# One-by-one convolution 
one-by-one convolutions allows you to shrink the number of channels and therefore save on computation in some networks. The effect of a one-by-one convolution is it just has nonlinearity. It allows you to learn a more complex function of your network by adding another layer,
bottleneck layer can be shrinked down the representation size significantly, and it doesn't seem to hurt the performance, but saves you a lot of computation.

the depthwise separable convolution has two steps. You're going to first use a depthwise convolution, followed by a pointwise convolution. It is these two steps which together make up this depthwise separable convolution.
