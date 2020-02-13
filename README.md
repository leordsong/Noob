# NerualNetwork
Python Implementations of neural network

## One Hidden Layer Nerual Network
Given a simple 1 hidden layer neural network of N input and L hidden units and
sigmoidal activation everywhere.<br/>
Then we have:

\begin{align}
f(\vect{x}, \theta) &= 
\sigma(\vect{w}^{\intercal}
\sigma(\vect{W}^{(1)\intercal}
 \vect{x} + \vect{b}^{(1)}) + b)  \\
\end{align}

<br/>

The partial derivative for an output weight wi between hidden unit i and the
output unit is given by

\begin{align}
\frac{\partial J(\vect{x})}{\partial w_{i}}
&= \frac{\partial(f(\vect{x}) - y)^2}{\partial w_{i}} \\
&= 2(f(\vect{x})-y) \frac{\partial f(\vect{x})}{\partial w_{i}} \\
&= 2(f(\vect{x})-y)f(\vect{x})(1 - f(\vect{x}))
\frac{\partial \sum_{l=1}^{L}w_lh_l + b}
{\partial w_{i}} \\
&= 2(f(\vect{x})-y)f(\vect{x})(1 - f(\vect{x})) h_i\\
\end{align}

<br/>

Similiarly, for the output bias weight b, it is given by

\begin{align}
\frac{\partial J(\vect{x})}{\partial b}
&= 2(f(\vect{x})-y)f(\vect{x})(1 - f(\vect{x}))\\
\end{align}
