## Why sequential models
Work best with sequential data as:

Speech Recognition
Music Generation
Sentiment Classification
DNA Sequence Analysis
Machine Translation
Video Activity Recognition
Name Entity Recognition
The given inputs may or may not be of the same kind or even have the same length.

## Notation
Given an input x we would have x<i> as the ith word in the sentence, for example.Tx denote how long the sequence is. In named entity recognition we could have the same length in the output denoted by Ty where each y<i> denotes a given output.  A common way to vectorize the given input  x<i> is using one hot encoded vectors. In this encoding each position of the vector represent a word.


## Recurrent Neural Network Model
A limitation of this type of network is that only uses information from the past of the sequences.


Different types of RNNs
Tx and Ty might not always be the same.

## Vanishing gradients with RNNs
In long sequences we have the same problem as very deep neural networks where the gradient can either explode or vanish. The exploding gradients problem is rather rare and possibly easier to address as we could simply clip it to stop it from growing. The vanishing on other hand can be partially solved with good parameter initialization but will keep happening in long structures. To address this problem other units have been developed.

## Bidirectional RNN
Bidirectional RNNs lets you take information from both earlier and later in the sequence. An output for a given time step is due the forward pass and the backward pass combined as we can see in the equation
