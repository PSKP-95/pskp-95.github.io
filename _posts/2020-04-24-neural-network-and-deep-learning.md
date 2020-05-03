---
layout: post
title: Neural Network and Deep Learning
description: Basics of feedforward neural networks. notes of coursera course 'neural network and deep learning' by andrew ng
keywords: Coursera, AI, ML, machine learning, deep learning, andrew ng, deeplearning.ai
author: Parikshit Patil
thumbnail: https://pskp-95.github.io/public/images/course1_dl.png
---

![Human neuron and artificial neuron](https://pskp-95.github.io/public/images/neuron_and_neuron.png)

Artificial Neural Networks are inspired by Human Neural System. In human neural system, axons of one neuron are connected with dendrites of another and they are regulating electric signal by using chemicals. This is higher level working of human neuron. number of these neurons are used for complex decision making. It's just intro :-).

## Logistic Regression

### Problem

We have to find **is there a cat in image or not**. This is binary classification because either cat is present or not in image (1 or 0).

![Cat or Not](https://pskp-95.github.io/public/images/cat_binary.png)

Before we start learning logistic regression, lets see how image is stored and presented in computer because we have to work on these images. Images are **continuous signals in space** but storing and processing continuous signal is too hard :-(. So images are discretized and then stored and processed in computers. images are stored in 3D matrix with 3 channels i.e. red, green, blue (RGB).

![Image Representation](https://pskp-95.github.io/public/images/image_repre.png)

As shown in above image, images are shored in 3D matrix. But for training on logistic regression, we need vector as a input. So We will roll out this image into **long column vector**. given image is **64x64x3** size where \\(1^{st}\\) 64 is height and \\(2^{nd}\\) width of image. Actually it depends on situation. for example, in numpy **img.shape** gives first dimension as height of image while mentioning resolution of image, we do opposite like **1920x1080**. 

**So the problem**: Given a cat picture \\(X\\), we want probability of cat in image i.e. \\(\hat{y} = P(y=1&#124;x)\\).

### Notation

We will see all notation in basic neural network. some of them are not used in logistic regression but we need them as we progress in article.

#### Sizes

 - \\(m\\): number of examples in dataset
 - \\(n_x\\): input size
 - \\(n_y\\): output size
 - \\(n^{[l]}_h\\): Number of hidden units in \\(l^{th}\\) layer.
 - \\(L\\): number of layers in metwork.

#### Objects

 - \\(X \in \R^{n_x \times m}\\) is the input matrix
 - \\(x^i \in \R^{n_x}\\) is the \\(i^{th}\\) example represented as a **column vector**
 - \\(Y \in \R^{n_y \times m}\\) is the label matrix or actual output while training
 - \\(y^{(i)} \in \R^{n_y}\\) is the output label for the \\(i^{th}\\) example
 - \\(W^{[l]} \in \R^{number \space of units \space in\space next\space layer \times number\space of\space units\space in\space previous\space layer}\\) is the weight matrix of \\(l^{th}\\) layer
 - \\(b^{[l]} \in \R^{number\space of\space units\space in\space next\space layer} \\) is the bias vector in the \\(l^{th}\\) layer
 - \\(\hat{y} \in \R^{n_y} \\) is the predicted output vector. Also denoted by \\(a^{[L]}\\). 

### Algorithm

Given \\(X\\), we want \\(\hat{y} = P(y=1&#124;0)\\). As below image, we will give input as rolled image and output will be 1 if cat in image else 0. in image, \\( w_1, w_2, ..., w_n \\) are weights and \\(b\\) is bias. These are learning parameters means we will learn them while training phase.

![Artificial Neuron](https://pskp-95.github.io/public/images/ar_neuron.webp)

if we want to explain above image in one line then this line will be like this **weighted sum over input is added to bias and whatever we got, pass it to activation function**. Here weighted sum means multiply weights to respective input i.e. \\(x_i * w_i\\) and add them. Lets consider \\(Z\\) as a sum of weighted sum of input and bias.
\\[
Z = b + \sum \limits_{i=1}^n x_i \times w_i 
\\]


#### Activation Function

If you are familiar with linear regression then you might saw above image except **activation function**. In linear regression, we are finding real value i.e. \\(Z\\) but here we want probability of cat in image then it must be in \\([0,1]\\). And also it introduces **non-linearity** to the output \\(Z\\). We will use **sigmoid activation function** in this turtorial. There are other activations functions are also available like ReLU, Leaky ReLU, tanh etc.

\\[
 \sigma(Z) = \frac{1}{1+e^{-Z}}
\\]


We need derivative of sigmoid function while sending error backward. So lets find now.

\\[
\begin{align*}
\sigma'(Z) &= \frac{d}{dZ} . \frac{1}{1 + e^{-Z}} \\\\
&= \frac{d}{dZ}(1+e^{-Z})^{-1} \\\\
&= -(1+e^{-Z})^{-2}(-e^{-Z}) \\\\
&= \frac{e^{-Z}}{(1+e^{-Z})^2} \\\\
&= \frac{1}{1+e^{-Z}} (1+\frac{1}{1+e^{-Z}}) \\\\
&= \sigma(Z).(1-\sigma(Z))
\end{align*}
\\]

When you plot sigmoid function, you will find graph shown below. 

![Sigmoid function](https://pskp-95.github.io/public/images/sigmoid.jpg)

#### Training

Weights \\(W^{[l]}\\) are initialized randomly to eliminate symmetry problem which arises when we initialize weights to zeros. Biases can be initialized to zeros. **Basic steps are** **1.** Ininitialize weights and biases **2.** forward pass means find \\(\hat{y}\\) using input and weights, biases **3.** compare \\(\hat{y}\\) with \\(Y\\) i.e. predicted output with actual output and then backpropagate errors to make changes appropriately in weights and biases.

Lets take example. We have dataset with 10000 (\\(m\\)) images of with cat and without cat in it. when every image of size \\(64 \times 64\times 3\\) rolled out forms 12228 (\\(n_x\\)) features long vector. We got imput \\(X^{n_x \times m}\\). Now label (Actual Output) for these examples is \\(Y^{1 \times m}\\). 

Let's see code for **initializing and forward pass.**

```python
import numpy as np

# suppose we have imput and output with us.
print(X.shape)
print(Y.shape)
```

##### Output

<div style="background-color:black;color:white;width:100%;padding-left:10px">
(12228, 10000) <br>
(1, 10000)
</div><br>

```python
# initialize weights and biases
W = np.random.rand(12228, 1) # input * output
b = np.zeros(1,)

# learning rate
alpha = 0.001

# Now forward pass
Z = np.dot(W.T, X) + b

def sigmoid(X):
    return 1 / (1 + np.exp(- X))

y_hat = sigmoid(Z)
```

Now time for **Back Propagation**. But it is too risky to say back propagation because in logistic regression, error is not sent back to another layer (:-) Only Single layer). For computing loss in single training example, we can formulate loss (error) function as

\\[
    L(\hat{y},y) = -(y\log({\hat{y}}) + (1-y)\log({1-\hat{y}}))
\\]

We can also use our loss function used in linear regression \\( L(\hat{y},y) = \frac{1}{2} (\hat{y} - y)^2\\) **but this loss function is non-convex** So its too hard to find global minima.

Until now we defined loss function which can tell us that **How good a particular example doing?** But we want to find **How good all examples doing?**. For that purpose we need **cost function** and it can be defined as

\\[
J(W,b) =  \frac{1}{m} \sum \limits_{i=1}^m L(\hat{y}^{i},y^{i})
\\]

##### Gradient Descent

Now only remaining job is to send loss backward and update weights and biases appropriately. For updating this parameters, we have to find cost w.r.t. weights and biases. Like below

\\[
\begin{align*}
W &= W - \alpha * \frac{\delta{J}(W,b)}{\delta{W}} \\\\
b &= b - \alpha * \frac{\delta{J}(W,b)}{\delta{b}}
\end{align*}
\\]

In above equation, \\(\alpha\\) is **learning rate**.

>The amount that the weights are updated during training is referred to as the step size or the **learning rate.** Specifically, the learning rate is a configurable **hyperparameter** used in the training of neural networks that has a small positive value, often in the range between **0.0 and 1.0**.

To find \\(\frac{\delta{J}(W,b)}{\delta{W}}\\) and \\(\frac{\delta{J}(W,b)}{\delta{b}}\\), we need to expand equation of \\(J(W,b)\\).

\\[
\begin{align*}
J(W,b) &= \frac{1}{m} \sum \limits_{i=1}^m L(\hat{y}^{i},y^{i}) \\\\

&= -\frac{1}{m} \sum \limits_{i=1}^m (y^{i}\log({\hat{y}^{i}}) + (1-y^{i})\log({1-\hat{y}^{i}})) \\\\

&= -\frac{1}{m} \sum \limits_{i=1}^m (y^{i}\log(\sigma{(z^{i})}) + (1-y^{i})\log({1-\sigma{(z^{i})}}))

\end{align*}
\\]

It's time to find gradients w.r.t. weights and biases. **currently, call \\(\hat{y}\\) as \\(a\\).**

\\[
\begin{align*}
\frac{\delta{J}( W,b )}{\delta{a}} &= -(\frac{y}{a} - \frac{1-y}{1-a}) \\\\

&= -\frac{y}{a} + \frac{1-y}{1-a} \\\\

\frac{\delta{J}( W,b )}{\delta{Z}} &= -(\frac{y}{\sigma{(Z)}} \sigma'{(Z)} + \frac{(1-y)}{(1-\sigma{(Z)})} \sigma'{(Z)})\\\\

&=-(\frac{y}{\sigma{(Z)}} \sigma(Z).(1-\sigma(Z)) + \frac{(1-y)}{(1-\sigma{(Z)})} \sigma(Z).(1-\sigma(Z)))\\\\

&=-(y.(1-\sigma(Z)) + (1-y)\sigma(Z))\\\\
&=a-y
\end{align*}
\\]

Now we will use above 2 equations to find \\(\frac{\delta{J}(W,b)}{\delta{W}}\\) and \\(\frac{\delta{J}(W,b)}{\delta{b}}\\)

<notextile>
\\[
\begin{align*}
\frac{\delta{J}( W,b )}{\delta{W}} &= \frac{\delta{J}{(W,b)}}{\delta{Z}} \frac{\delta{Z}}{\delta{W}} //

&=(a-y) . X^T //

\frac{\delta{J}(W,b)}{\delta{b}} &= \frac{\delta{J}{(W,b)}}{\delta{Z}} \frac{\delta{Z}}{\delta{b}} //

&=a-y 
\end{align*}
\\]
</notextile>
<notextile>
\[
\begin{align*}
\frac{\delta{J}( W,b )}{\delta{W}} &= \frac{\delta{J}{(W,b)}}{\delta{Z}} \frac{\delta{Z}}{\delta{W}} //

&=(a-y) . X^T //

\frac{\delta{J}(W,b)}{\delta{b}} &= \frac{\delta{J}{(W,b)}}{\delta{Z}} \frac{\delta{Z}}{\delta{b}} //

&=a-y 
\end{align*}
\]
</notextile>

{% raw %}
\\[
\begin{align*}
\frac{\delta{J}( W,b )}{\delta{W}} &= \frac{\delta{J}{(W,b)}}{\delta{Z}} \frac{\delta{Z}}{\delta{W}} //

&=(a-y) . X^T //

\frac{\delta{J}(W,b)}{\delta{b}} &= \frac{\delta{J}{(W,b)}}{\delta{Z}} \frac{\delta{Z}}{\delta{b}} //

&=a-y 
\end{align*}
\\]
{% endraw %}
Finally done :-). Time for code.

```python
# Find cost

cost = -(1/m)*(Y*np.log(y_hat)+(1-Y)*np.log(1-y_hat))

dw = np.dot((y_hat - y),X.T)
db = y_hat - y

# update w and b
W = W - alpha * dw
b = b - alpha * db
```

## References
[1] [All notations reference](https://d3c33hcgiwev3.cloudfront.net/_106ac679d8102f2bee614cc67e9e5212_deep-learning-notation.pdf?Expires=1588550400&Signature=Wp2N8mhdPu~oh8SEoq-jjl-nb0p6O9E-P3mDH3GNhxzu~p8AU4UQ32MiFIE~S0Y31uQysTlWD6VR0cOvmE3rZVUGL94U1TtitKo9KAF72GMCBrgP7yFnkuS730lkKzs2jmHAZV09hjvqCyuHcpu6DyoPUrCeMC1wJCVgItmsMqo_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

