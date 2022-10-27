---
layout: info
title: Test for math
permalink: /publication/test/
---

$\mathbf{s}_{r}^{1:M}$


### Math2


incline math: $\mathbf{s}_{r}^{1:M}$

 $\mathbf{s}_{h}^{1:M}$

black math:


$$
\begin{align*}    \mathcal{L}_{CVAE} & \simeq \frac{1}{N} \sum_{i=1}^{N} \log \big( p_{\theta}(\mathbf{y} \mid \hat{\mathbf{z}_{i}}, \mathbf{x})\big) - D_{KL}\big(q_{\phi}(\mathbf{z} \mid \mathbf{x}, \mathbf{y} )\lVert p_{\theta}(\mathbf{z} \mid \mathbf{x})\big), \\    & \hat{\mathbf{z}}_i \sim q_{\phi}(\mathbf{z} \mid \mathbf{x}, \mathbf{y})     \end{align*}
$$

We collect a data set of human-robot interactions for training and validation. Each demonstration consists of the robot end-effector positions in the cartesian space $\mathbf{s}_{r}^{1:M}$, and the positions of the human hand $\mathbf{s}_{h}^{1:M}$, with $M$ time steps. The test data is generated online.


To implement an LSTM-based CVAE for HRI, the input of the model is  $\mathbf{x} = [\mathbf{x}_{h}^{1:t}, \mathbf{x}_{r}^{1:t}]$​​​​​​​ and the output is the control signal $\mathbf{y} = \ddot{\mathbf{s}}_r^{t+1:t+T}$​, where $t \in [2 {.}\,{.} M-T]$. 
