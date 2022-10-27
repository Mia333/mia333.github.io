---
layout: info
title: Test for math
permalink: /publication/test/
---

$\mathbf{s}_{r}^{1:M}$

$\mathbf{s}_{aug}^{0:T} = \lambda \mathbf{s}_{a}^{0:T} + (1-\lambda) \mathbf{s}_{b}^{0:T}$
 math

### Math2


We collect a data set of human-robot interactions for training and validation. Each demonstration consists of the robot end-effector positions in the cartesian space $\mathbf{s}_{r}^{1:M}$, and the positions of the human hand $\mathbf{s}_{h}^{1:M}$, with $M$ time steps. The test data is generated online.


To implement an LSTM-based CVAE for HRI, the input of the model is  and the output is the control signal $\mathbf{y} = \ddot{\mathbf{s}}_r^{t+1:t+T}$​, where $t \in [2 \dots M-T]$. $\mathbf{x}^t_h$ and $\mathbf{x}^t_r$​​​, consisting of the position, velocity, and acceleration, are the state of human and robot past trajectory from time step one to the current time step $t$​​​​​​​​ (see the notation in Fig. 1). The long output of the model from time step $t$ to $T$ is only used in the training and validation processes, resulting in more accurate predictions. However, we only use $\ddot{s}_r^{t+1}$ for testing.


Given a recognition model $q_\phi (\mathbf{z} | \mathbf{x}, \mathbf{y})$​, a generation model $p_\theta(\mathbf{y} | \mathbf{z}, \mathbf{x})$​, and a conditional prior model $p(\mathbf{z}|\mathbf{x})$, we approximates the evidence lower bound (ELBO) of the CVAE [[3]](#3):


<!-- $$
\begin{align*}    \mathcal{L}_{CVAE} & \simeq \frac{1}{N} \sum_{i=1}^{N} \log \big( p_{\theta}(\mathbf{y} \mid \hat{\mathbf{z}_{i}}, \mathbf{x})\big) - D_{KL}\big(q_{\phi}(\mathbf{z} \mid \mathbf{x}, \mathbf{y} )\lVert p_{\theta}(\mathbf{z} \mid \mathbf{x})\big), \\    & \hat{\mathbf{z}}_i \sim q_{\phi}(\mathbf{z} \mid \mathbf{x}, \mathbf{y})     \end{align*}
$$ -->

<p align="center">
<img src="/assets/cvae4hri/eq_Lcvae.png">
</p>

where $N$​ is the number of samples. Given $\mathbf{x}$, $\mathbf{z}$ is able to model multiple modes in conditional distribution of the output $\mathbf{y}$.

<p align="center">
<img src="/assets/cvae4hri/lstm-cvae_framework.png">
<em>Figure 1. The proposed model.</em>
</p>
 
 
 *  Data augmentation for sequential data using MIXUP

Demonstration collection on a real robot is inefficient and time-consuming. Therefore, we use mixup [[2]](#2) for data augmentation. In the original mixup method, discrete datapoints are augmented. Given demonstration $a$ and $b$, we adapt this technique to our specific sequential interaction dataset for both human and robot trajectories, 
<!-- $\mathbf{s}_{aug}^{0:T} = \lambda \mathbf{s}_{a}^{0:T} + (1-\lambda) \mathbf{s}_{b}^{0:T}$,  -->
<p align="center">
<img src="/assets/cvae4hri/eq_mixup.png">
</p>
 
where $\lambda$ is randomly sampled. 
