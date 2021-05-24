 ## What is neural network?
 X, y, w, b -----> Z = w.T * X + b ------> A = sigmoid(Z) -----> loss(A, y) 
 
 file logisticregressionNNminset.py is an example that shows a simple mindset of neural network implementation 
 
 
 single hiden layer NN
<img src="https://user-images.githubusercontent.com/64529936/119261066-f6ebaa80-bbd5-11eb-962f-0641b5a61f57.png" width="300" height="300">

input layer input featurs to hiden layers

first node in hiden layers computes like z1 = w1.T * x1 + b1,  a1 = sigmoid(z1)  and second node and the rest computed in the same way
(Z is a column vector, w is a (nxn) matric, b is a (nx1) array )
the horizontally the matrix A goes over different training examples and vertically the different indices in the matrix A corresponds to different hidden units.
  
forward prppagation:  
  
z[1]= w[1]x +b[1] ---
a[1] =sigmid[z[1]] ---
z[2] w[2]a[1]+ b[2] ---
a[2] = simoid[z[2]] 

activation functions

a = sigmoid(z) =. 1/1+exp(-z).      ------ binary calssificaton y = {0,1}

a = tanh(z)  = exp(z)-exp(-z)/exp(z)+exp(-z) -------- -1<y<1

                                  /
a = max{0,z}.  --->relu =  ------/.      ------> computation is faster , learning nn is faster becous of slop of relu or leaky relu.

backward propagation:

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
 
 

