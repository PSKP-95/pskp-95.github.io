---
layout: post
title: Tour to the land of activation functions 
description: pros and cons of each activation function with detailed description like mathematical formulae and it's derivative
keywords: activation functions in detail,ReLU, Leaky ReLU, approximates identity near the origin, neural network,AI,ML,machine learning,deep learning
author: Parikshit Patil
thumbnail: /public/images/relu_icon.png
---

![Activation Function Meme](/public/images/activation_meme.jpg)

If you are unfamiliar to basics of neural network and backpropagation in neural network, I recommend to check below articles. These articles increase your confidence while reading this article. You can find all mathematical basic concepts about feedforward neural network in these articles.

<div class="preview">
    <div class="left" onclick="location.href='/2020/04/24/neural-network-and-deep-learning/'">
      <div class="head">
        <h2>Neural Network and Deep Learning (Part 1)</h2>
      </div>
      <div class="detail">
        <p>Basics of feedforward neural networks. notes of coursera course 'neural network and deep learning' by andrew ng</p>
      </div>
      <div class="link">
        <p><i class="fa fa-link" aria-hidden="true"></i>
 pskp-95.github.io</p>
      </div>
    </div>
    <div class="right">
      <img align='right' src="/public/images/course1_dl.png" alt="">
    </div>
</div>

<div class="preview">
    <div class="left" onclick="location.href='/2020/04/26/neural-network-and-deep-learning-1/'">
      <div class="head">
        <h2>Neural Network and Deep Learning (Part 2)</h2>
      </div>
      <div class="detail">
        <p>Basics of feedforward neural networks. Generic implemention for L layer deep neural network in numpy.</p>
      </div>
      <div class="link">
        <p><i class="fa fa-link" aria-hidden="true"></i>
 pskp-95.github.io</p>
      </div>
    </div>
    <div class="right">
      <img align='right' src="/public/images/course1_dl.png" alt="">
    </div>
</div>

<div class="preview" onclick="location.href='/2020/04/28/neural-network-and-deep-learning-2/'">
    <div class="left">
      <div class="head">
        <h2>Neural Network and Deep Learning (Part 3)</h2>
      </div>
      <div class="detail">
        <p>Multiclass classification using deep neural network in numpy only. Also MNIST handwritten digit classification in numpy.</p>
      </div>
      <div class="link">
        <p><i class="fa fa-link" aria-hidden="true"></i>
 pskp-95.github.io</p>
      </div>
    </div>
    <div class="right">
      <img align='right' src="/public/images/course1_dl.png" alt="">
    </div>
</div>

Before we start our journey <i class="fas fa-plane-departure"></i> let's find `what are the properties of activation function are neccessary to become good activation function?`

## Characteristics

 - **`Nonlinear:`** When the activation function is non-linear, then a two-layer neural network can be proven to be a universal function approximator
> **a universal function** is a computable function capable of calculating any other computable function

   If we use **Linear functions** throughout in network then network is same as **perceptron (single layer network)**
 ![Linear and Non Linear Functions](/public/images/linear_non.png)
 *Source: study.com*
   
 - **`Range:`** When range is **finite**, gradient based optimization methods are more stable because it limits the weights. When range is **infinite**, gradient based optimization methods are more efficient but for smaller learning rate. because weights updation don't have limit of activation function. You can refer \\(1^{st}\\) article in above list. You will find, weight updation is depends on activation function also.
  >**Range and Domain**: The domain of a function \\(f(x)\\) is the set of all values for which the function is defined, and the range of the function is the set of all values that \\(f\\) takes. 

    For example take function \\(f(x)=\sin{(x)}\\) which is sine function. Its range is \\([-1,1]\\) and domain is \\((-\infty,+\infty)\\)
  ![Sine Wave](/public/images/sine_wave.svg)
  *Source: mathisfun.com*

 - **`Continuously differentiable:`** A continuously differentiable function \\(f(x)\\) is a function whose derivative function \\(f'(x)\\) is also continuous in it's domain. I recommend to check [Youtube: Continuity Basic Introduction, Point, Infinite, & Jump Discontinuity, Removable & Nonremovable](https://www.youtube.com/watch?v=joewRl1CTL8)
  ![Sine Wave](/public/images/continuity.png)
  *Source: calcworkshop.com*
  
    In below image, function is **binary step function** and it is discontinuous at \\(x=0\\) and it is **jump discontinuity**. As it is not differentiable at \\(x=0\\), so gradient-based methods can make no progress with it
  ![Binary Step Function](/public/images/step_binary.png)
  *Source: calcworkshop.com*
 - **`Monotonic:`** 
 >In calculus, a function \\(f\\)  defined on a subset of the real numbers with real values is called monotonic if and only if it is either entirely non-increasing, or entirely non-decreasing.
 
    **Identity Function** is monotonic function (\\(f(x)=x\\)) and \\(f(x)=\sin{(x)}\\) is non-monotonic function.
  ![Monotonic Function](/public/images/monotonic.png)
  **When the activation function is monotonic, the error surface associated with a single-layer model is guaranteed to be convex.**
 - **`Monotonic Derivative:`** Smooth functions with a monotonic derivative have been shown to generalize better in some cases. I think its because of local minima problem. While training sometimes network stuck at local minima instead of global minima

    ![Monotonic Function](/public/images/local_minima.png)
    *Source: researchgate.net*
    
 - **`Approximates identity near the origin:`** Usually, the weights \\(W\\) and bias \\(b\\) are initialized with values close to zero by the gradient descent method. Consequently, \\(WX^T + b\\) or in our case \\(WX + b\\) (Check Above Series) will be close to zero. 
 >If \\(f\\) approximates the identity function near zero, its gradient will be approximately equal to its input. 

   In other words, \\(\partial{f} ≈ WX^T + b ⇐⇒ WX^T + b ≈ 0\\). In terms of the gradient descend, it is a strong gradient which helps the training algorithm to converge faster.

## Activation Functions

From onward \\(f(x)\\) is equation of activation function and \\(f'(x)\\) is derivative of that activation function which is required during backpropagation. We will see most used activation function and you can find others on wikipedia page [Link](#references). All function graphs are taken from book named **Guide to Convolutional Neural Networks: A Practical Application to Traffic-Sign Detection and Classification** written by **Aghdam, Hamed Habibi** and **Heravi, Elnaz Jahani**

### 1. Sigmoid

<div class="scrollable">
<notextile>
\[
\begin{align*}
f(x) &= \frac{1}{1+e^{-x}} \\
f'(x) &= f(x)(1-f(x))
\end{align*}
\]
</notextile>
</div>

![Sigmoid Function](/public/images/activation-functions/sigmoid.png)

#### Pros:

- It is nonlinear, so it can be used to activate hidden layers in a neural network
- It is differentiable everywhere, so gradient-based backpropagation can be used with it

#### Cons:

- The gradient for inputs that are far from the origin is near zero, so gradient-based learning is slow for saturated neurons using sigmoid i.e. **vanishing gradients problem**
- When used as the final activation in a classifier, the sum of all classes doesn’t necessarily total 1
- For these reasons, sigmoid activation function is not used in deep architectures since training the network become nearly impossible

#### Summary

<div class="scrollable">
<notextile>
<table>
  <tr>
    <th>Range</th>
    <th>Order of Continuity</th>
    <th>Monotonic</th>
    <th>Monotonic Derivative</th>
    <th>Approximates Identity near the origin</th>
  </tr>
  <tr>
    <td>\((0,1)\)</td>
    <td>\(C^{\infty}\)</td>
    <td><i class="fa fa-check" style="color:green" aria-hidden="true"></i></td>
    <td><i class="fa fa-times" style="color:red" aria-hidden="true"></i></td>
    <td><i class="fa fa-times" style="color:red" aria-hidden="true"></i></td>
  </tr>
</table>
</notextile>
</div>

### 2. Hyperbolic Tangent

<div class="scrollable">
<notextile>
\[
\begin{align*}
f(x) &= \tanh{x} \\
 &= \frac{e^{x} - e^{-x}}{e^{x}+e^{-x}} \\
f'(x) &= 1-(f(x))^2
\end{align*}
\]
</notextile>
</div>

![Hyperbolic Tangent Function](/public/images/activation-functions/tanh.png)

The hyperbolic tangent activation function is in fact a rescaled version ofthe sigmoid function.

#### Pros:
- It is nonlinear, so it can be used to activate hidden layers in a neural network
- It is differentiable everywhere, so gradient-based backpropagation can be used with it
- It is preferred over the sigmoid function because **it approximates the identity function near origin**

#### Cons:

- As \\(\|x\|\\) increases, it may suffer from **vanishing gradient problems** like sigmoid.
- When used as the final activation in a classifier, the sum of all classes doesn’t necessarily total 1

#### Summary

<div class="scrollable">
<notextile>
<table>
  <tr>
    <th>Range</th>
    <th>Order of Continuity</th>
    <th>Monotonic</th>
    <th>Monotonic Derivative</th>
    <th>Approximates Identity near the origin</th>
  </tr>
  <tr>
    <td>\((-1,1)\)</td>
    <td>\(C^{\infty}\)</td>
    <td><i class="fa fa-check" style="color:green" aria-hidden="true"></i></td>
    <td><i class="fa fa-times" style="color:red" aria-hidden="true"></i></td>
    <td><i class="fa fa-check" style="color:green" aria-hidden="true"></i></td>
  </tr>
</table>
</notextile>
</div>

### 3. Rectified Linear Unit

<div class="scrollable">
<notextile>
\[
\begin{align*}
f(x)&= max(0,x) \\
      &=\begin{cases}
          0,        & \text{for } x\leq 0\\
          x,        & \text{for } x \gt 0
      \end{cases} \\
f'(x) &= \begin{cases}
          0,        & \text{for } x\leq 0\\
          1,        & \text{for } x \gt 0
      \end{cases}
\end{align*}
\]
</notextile>
</div>

![relu Function](/public/images/activation-functions/relu.png)

#### Pros:
- Computationaly very efficient
- Its derivative in R+ is always 1 and it does not saturate in R+ (No vanishing gradient problem)
- Good choice for **deep networks**
- Problem of dead neuron may affect learning but it makes more efficient at the time of inferece because we can remove these **dead neurons**.

#### Cons:
- Function does not approximate the identity function near origin.
- It may produce **dead neuron**. A dead neuron always return 0 for every sample in the dataset. This affects accuracy of model.
>This happens because the weight of dead neuron have been adjusted such that \\(WX^T + b\\) for the neuron is **always negative.**

#### Summary

<div class="scrollable">
<notextile>
<table>
  <tr>
    <th>Range</th>
    <th>Order of Continuity</th>
    <th>Monotonic</th>
    <th>Monotonic Derivative</th>
    <th>Approximates Identity near the origin</th>
  </tr>
  <tr>
    <td>\([0,\infty)\)</td>
    <td>\(C^{0}\)</td>
    <td><i class="fa fa-check" style="color:green" aria-hidden="true"></i></td>
    <td><i class="fa fa-check" style="color:green" aria-hidden="true"></i></td>
    <td><i class="fa fa-times" style="color:red" aria-hidden="true"></i></td>
  </tr>
</table>
</notextile>
</div>

### 4. Leaky Rectified Linear Unit

<div class="scrollable">
<notextile>
\[
\begin{align*}
f(x)&= 
      \begin{cases}
          0.01x,        & \text{for } x\leq 0\\
          x,        & \text{for } x \gt 0
      \end{cases} \\
f'(x) &= \begin{cases}
          0.01,        & \text{for } x\leq 0\\
          1,        & \text{for } x \gt 0
      \end{cases}
\end{align*}
\]
</notextile>
</div>

![leaky relu Function](/public/images/activation-functions/lrelu.png)

In practice, leaky ReLU and ReLU may produce similar results. This might be due to the fact that the positive region of these function is identical. 0.01 can be changed with other values between \\([0,1]\\)/

#### Pros:
- As its gradient does not vanish in negative region as opposed to ReLU, it solves the problem of dead neuron.

#### Cons:
- 

#### Summary

<div class="scrollable">
<notextile>
<table>
  <tr>
    <th>Range</th>
    <th>Order of Continuity</th>
    <th>Monotonic</th>
    <th>Monotonic Derivative</th>
    <th>Approximates Identity near the origin</th>
  </tr>
  <tr>
    <td>\((-\infty,+\infty)\)</td>
    <td>\(C^{\infty}\)</td>
    <td><i class="fa fa-check" style="color:green" aria-hidden="true"></i></td>
    <td><i class="fa fa-check" style="color:green" aria-hidden="true"></i></td>
    <td><i class="fa fa-times" style="color:red" aria-hidden="true"></i></td>
  </tr>
</table>
</notextile>
</div>

### 5. Parameterized Rectified Linear Unit

<div class="scrollable">
<notextile>
\[
\begin{align*}
f(x)&= 
      \begin{cases}
          \alpha x,        & \text{for } x\leq 0\\
          x,        & \text{for } x \gt 0
      \end{cases} \\
f'(x) &= \begin{cases}
          \alpha,        & \text{for } x\leq 0\\
          1,        & \text{for } x \gt 0
      \end{cases}
\end{align*}
\]
</notextile>
</div>

![Parameterized relu Function](/public/images/activation-functions/prelu.png)

It is same as Leaky ReLU but \\(\alpha\\) is **learnable parameter** can be learned from data.

#### Parameter Updation

To update \\(\alpha\\), we need gradient of activation function with respect to \\(\alpha\\).

<div class="scrollable">
<notextile>
\[
\begin{align*}
\partial{f(x)}&= 
      \begin{cases}
           x,        & \text{for } x\lt 0\\
          \alpha,        & \text{for } x \geq 0
      \end{cases} \\
\end{align*}
\]
</notextile>
</div>

#### Summary

<div class="scrollable">
<notextile>
<table>
  <tr>
    <th>Range</th>
    <th>Order of Continuity</th>
    <th>Monotonic</th>
    <th>Monotonic Derivative</th>
    <th>Approximates Identity near the origin</th>
  </tr>
  <tr>
    <td>\((-\infty,+\infty)\)</td>
    <td>\(C^{\infty}\)</td>
    <td><i class="fa fa-check" style="color:green" aria-hidden="true"></i></td>
    <td><i class="fa fa-check" style="color:green" aria-hidden="true"></i> if \(\alpha \geq 0\)</td>
    <td><i class="fa fa-check" style="color:green" aria-hidden="true"></i> if \(\alpha = 1\)</td>
  </tr>
</table>
</notextile>
</div>

### 6. Softsign

<div class="scrollable">
<notextile>
\[
\begin{align*}
Z^{[l]} &= W^{[l]}.a^{[l-1]} + b^{[l]} \\
a^{[l]} &= g^{[l]}{(Z^{[l]})}
\end{align*}
\]
</notextile>
</div>

![Softsign Function](/public/images/activation-functions/softsign.png)

#### Pros:
- The function is equal to zero at origin and its derivative at origin is equal to 1. Therefore, is approximates the identity function at origin
- Comparing the function and its derivative with hyperbolic tangent, we observe that it also saturates as \\(\|x\|\\) increases. However, **the saturation ratio of the softsign function is less than the hyperbolic tangent function which is a desirable property**
- gradient of the softsign function near origin drops with a greater ratio compared with the hyperbolic tangent.
- In terms of computational complexity, softsign requires less computation than the hyperbolic tangent function.

#### Cons:
- saturates as \\(\|x\|\\) increases

#### Summary

<div class="scrollable">
<notextile>
<table>
  <tr>
    <th>Range</th>
    <th>Order of Continuity</th>
    <th>Monotonic</th>
    <th>Monotonic Derivative</th>
    <th>Approximates Identity near the origin</th>
  </tr>
  <tr>
    <td>\((-1,1)\)</td>
    <td>\(C^{1}\)</td>
    <td><i class="fa fa-check" style="color:green" aria-hidden="true"></i></td>
    <td><i class="fa fa-times" style="color:red" aria-hidden="true"></i></td>
    <td><i class="fa fa-check" style="color:green" aria-hidden="true"></i></td>
  </tr>
</table>
</notextile>
</div>

### 7. Softplus

<div class="scrollable">
<notextile>
\[
\begin{align*}
f(x) &= \ln(1+e^x) \\
fi(x) &= \frac{1}{1+e^{-x}} 
\end{align*}
\]
</notextile>
</div>

![Softsign Function](/public/images/activation-functions/softplus.png)

#### Pros:
- In contrast to the ReLU which is not differentiable at origin, the softplus function is differentiable everywhere
- The derivative of the softplus function is the sigmoid function which means the range of derivative is \\([0,1]\\)

#### Cons:
- the derivative of softplus is also a smooth function which saturates as \\(\|x\|\\) increases

#### Summary

<div class="scrollable">
<notextile>
<table>
  <tr>
    <th>Range</th>
    <th>Order of Continuity</th>
    <th>Monotonic</th>
    <th>Monotonic Derivative</th>
    <th>Approximates Identity near the origin</th>
  </tr>
  <tr>
    <td>\((0,\infty)\)</td>
    <td>\(C^{\infty}\)</td>
    <td><i class="fa fa-check" style="color:green" aria-hidden="true"></i></td>
    <td><i class="fa fa-check" style="color:green" aria-hidden="true"></i></td>
    <td><i class="fa fa-times" style="color:red" aria-hidden="true"></i></td>
  </tr>
</table>
</notextile>
</div>

## References

[[1] Activation Functions - (wikipedia)](https://en.wikipedia.org/wiki/Activation_function)

[[2] Pros and Cons of Sigmoid Activation Function](https://www.quora.com/What-are-the-pros-and-the-cons-of-the-sigmoid-activation-function-in-deep-learning)