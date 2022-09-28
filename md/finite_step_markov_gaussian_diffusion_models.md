# Finite Step Unconditional Markov Diffusion Models

### Gaussian diffusion

For the particular case of Gaussian diffusion models, we assume

* the terminal distribution is normal
  
  $$
  q_{T|0} = q_{T}=p_{T} = \mathcal N({\bf x}_{T};{\bf 0}, {\bf 1}),
  $$

* forward transition process is Gaussian
  
  $$
  q_{t|t-1} = \mathcal N({\bf x}_{t}; \sqrt{1-\beta_{t}}{\bf x}_{t-1}, \beta_{t} {\bf 1})
  $$
  
  where $0<\beta_{t} \le 1$.

It is useful to introduce additional notations:

* $\alpha_{t} \equiv 1 - \beta_{t}$, and
* $\bar \alpha_{t} \equiv \prod_{t=1}^T \alpha_{t}$.

#### Reparameterization

Let ${\bf z}\sim \mathcal N({\bf 0}, {\bf 1})$, the forward process can be written as

$$
{\bf x}_{t} = \sqrt{1-\beta_{t}} {\bf x}_{t-1} + \sqrt{\beta_{t}} {\bf z} =\sqrt{\alpha_{t}} {\bf x}_{t-1} + \sqrt{\beta_{t}} {\bf z} .
$$

Iteratively apply the formula, we get

$$
{\bf x}_{t} = \sqrt{\bar \alpha_{t}} {\bf x}_0 + \sqrt{1-\bar \alpha_{t}} {\bf z}.
$$

> Thus, we can generate ${\bf x}_{t}$ for **any** $t$ without actually do the iterative calculations.
> There is a similar property for any Markov process, i.e. Feynman-Kac formula

#### Posterior is Gaussian

We can compute the posterior of the physical process using

$$
q_{t-1|t, 0} \propto q_{t|t-1, 0} q_{t-1 | 0}
$$

where the RHS is a product of Gaussians. One can show that the posterior is indeed Gaussian after doing an easy but lengthy calculation

$$
q_{t-1|t, 0} = \mathcal N({\bf x}_{t-1}; \tilde \mu_{t}({\bf x}_{t}, {\bf x}_0), \tilde \beta_{t} {\bf 1})
$$

where

$$
\tilde \mu_{t} = \frac{\sqrt\alpha_{t} (1 - \bar \alpha_{t-1}){\bf x}_{t} + \beta_{t} \sqrt{\bar \alpha_{t-1}}{\bf x}_0}{1-\bar\alpha_{t}} ,
$$

and

$$
\tilde \beta_{t} = \beta_{t} \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_{t}} .
$$

Express ${\bf x}_0$ in terms of ${\bf x}_{t}$ and noise, the mean simplies

$$
\tilde \mu_{t} ({\bf x}_{t}) = \alpha_{t}^{-\frac12 } \left( {\bf x}_{t} - \frac{\beta_{t}}{\sqrt{1-\bar \alpha_{t}}} {\bf z} \right).
$$

Note that if one naively inverts the forward process formula, one would get

$$
{\bf x}_{t-1} = \alpha_{t}^{-\frac12} \left( {\bf x}_{t} - \sqrt{\beta_{t}} {\bf z}\right)
$$

but the posterior mean is

$$
\tilde \mu_{t} ({\bf x}_{t}) = \alpha_{t}^{-\frac12 } \left( {\bf x}_{t} - \sqrt{\frac{1 - \alpha_{t}}{1-\bar \alpha_{t}}} \sqrt{\beta_{t}}{\bf z} \right)
$$

where

$$
0 < \sqrt{ \frac{1-\alpha_{t}}{1-\bar \alpha_{t}} } \le 1 .
$$

Thus, the posterior backward process is removing less noise except for $t=1$ than what one would naively do.

#### Variational Ansatz

Since the target distribution is Gaussian, it is a good idea to choose Gaussian distribution as the variational Ansatz

$$
p_{t-1|t} = \mathcal N ({\bf x}_{t-1}; \mu_{t}, \sigma^2_{t}{\bf 1} )
$$

where the model parameters are $\mu_{t}$ and $\sigma_{t}$. The variance will eventually contribute to learning rate; we will treat $\sigma_{t}$ as a hyperparameter instead of learning it from stochastic gradient descent. The only learnable parameter is then $\mu_{t}=\mu_{t}({\bf x}_{t}, t)$.

Recall the objective for each time step $t=2,\cdots,T$

$$
D_{t-1} = D_{\rm KL}(q_{t-1|t,0}|p_{t-1|t,0}) = \frac{1}{2\sigma^2_{t}} \Vert \mu_{t} - \tilde \mu_{t} \Vert^2 + {\rm const.}
$$

where we used the KL-divergence between two Gaussian distributions.

Clearly, we have the exact solution

$$
\frac{\delta D_{t-1}[\mu_{t}]}{\delta \mu_{t}} = 0 \Rightarrow \mu_{t} = \tilde \mu_{t},
$$

and therefore,

$$
\mu_{t} ({\bf x}_{t}, t) = \alpha_{t}^{-\frac12 } \left( {\bf x}_{t} - \frac{\beta_{t}}{\sqrt{1-\bar \alpha_{t}}} {\bf z}_{t} \right)
$$

where ${\bf z}_{t}\sim\mathcal N(0, {\bf 1})$ is the noise that generated ${\bf x}_{t}$ from ${\bf x}_0$.

Next, we reparameterize $\mu_{t}$ to separate the *explicit* dependency of ${\bf x}_{t}$ and $t$ and let the model only focus on the *implicit* dependencies, i.e.

$$
\mu_{t} ({\bf x}_{t}, t) = \alpha_{t}^{-\frac12 } \left( {\bf x}_{t} - \frac{\beta_{t}}{\sqrt{1-\bar \alpha_{t}}} {\bf z}({\bf x}_{t}, t) \right).
$$

Finally, the objective becomes

$$
D_{t-1} = \frac{\beta_{t}^2}{2\alpha_{t}(1-\bar\alpha_{t})\sigma_{t}^2} \Vert {\bf z}_{t} -{\bf z}({\bf x}_{t},t)\Vert^2
$$

where ${\bf z}({\bf x}_{t}, t)$ is the model output.

It can also be shown that

$$
L_0 = \frac{\beta_1}{2\alpha_1\sigma_1^2} \Vert {\bf z}_1 -{\bf z}({\bf x}_0,1)\Vert^2
$$

where $\bar \alpha_1 = \alpha_1 = 1 -\beta_1$.

Thus, the generic loss term for $0 < t < T$ is

$$
L_{t-1} = \frac{\beta_{t}^2}{2\alpha_{t}(1-\bar\alpha_{t})\sigma_{t}^2} \Vert {\bf z}_{t} -{\bf z}({\bf x}_{t},t)\Vert^2.
$$

Note that we still have the freedom to choose $\sigma_{t}$ that controls the importance of each step. But in the literature, they usually take a *heuristic* approach by ignoring the weight factor keeping only the $\ell_2$ loss.

> **Question** What if one trains the models using different choices of $\sigma_t$?

#### Sampling the backward process

During training, the model learned the backward transition distribution

$$
p_{t-1|t} = \mathcal N \left ({\bf x}_{t-1}; \alpha_{t}^{-\frac12 } \left( {\bf x}_{t} - \frac{\beta_{t}}{\sqrt{1-\bar \alpha_{t}}} {\bf z}({\bf x}_{t}, t) \right), \sigma^2_{t}{\bf 1} \right).
$$

The backward iteration is essentially sampling and calculating

$$
{\bf x}_{t-1} = \alpha_{t}^{-\frac12 } \left( {\bf x}_{t} - \frac{\beta_{t}}{\sqrt{1-\bar \alpha_{t}}} {\bf z}({\bf x}_{t}, t) \right) + \sigma_{t} {\bf z}
$$

where ${\bf z} \sim \mathcal N({\bf 0}, {\bf 1})$.

> **Question** What happens if one tries to reconstruct ${\bf x}_0$ using the incorrect, naive formula?

#### Training and inference algorithms

##### Training

* Sample
  * ${\bf x}_0 \sim q_0$
  * $t \sim {\sf Uniform}(1,\cdots,T)$
  * ${\bf z}_{t} \sim \mathcal N({\bf 0}, {\bf 1})$
* Construct ${\bf x}_{t}$
* Feed ${\bf x}_{t}, t$ to model
* Minimize $L_{t-1}$

##### Inference

* Sample $x_{T} \sim \mathcal N({\bf 0}, {\bf 1})$
* Loop $t = T,\cdots, 1$
  * Sample ${\bf z} \sim \mathcal N({\bf 0}, {\bf 1})$
  * Compute ${\bf x}_{t-1}$
* Return ${\bf x}_0$

The reconstruction formula is given in previous section "sampling the backward process."