---
layout: info
title: Test for math
permalink: /publication/test/
---

$\mathbf{s}_{r}^{1:M}$


### Math2


incline math: $\mathbf{s}_{r}^{1:M}$


black math:


$$
\begin{align*}    \mathcal{L}_{CVAE} & \simeq \frac{1}{N} \sum_{i=1}^{N} \log \big( p_{\theta}(\mathbf{y} \mid \hat{\mathbf{z}_{i}}, \mathbf{x})\big) - D_{KL}\big(q_{\phi}(\mathbf{z} \mid \mathbf{x}, \mathbf{y} )\lVert p_{\theta}(\mathbf{z} \mid \mathbf{x})\big), \\    & \hat{\mathbf{z}}_i \sim q_{\phi}(\mathbf{z} \mid \mathbf{x}, \mathbf{y})     \end{align*}
$$

We collect a data set of human-robot interactions for training and validation. Each demonstration consists of the robot end-effector positions in the cartesian space $\mathbf{s}_{r}^{1:M}$ and the positions of the human hand with $M$ time steps. The test data is generated online.

