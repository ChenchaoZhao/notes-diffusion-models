# Finite Step Unconditional Markov Diffusion Models

Let $p( \cdot | \theta )$ be the parametric model that models data $x_0 \sim q_0$, then we can optimize $\theta$ by maximize the likelihood $p(x_0 | \theta)$ or equivalently

$$
\max_\theta \mathbb E_{x_0 \sim q_0} \log p(x_0|\theta).
$$

From now on, we use $q$'s to denote the forward physical distributions and $p$'s the backward variational ansatz. The parameters are implied in $p(\cdot|\theta) \equiv p(\cdot)$.

Let $T > 1$ be the diffusion steps, the joint density of forward process $q(x_{0, 1, \cdots T})$ can be expanded sequentially if the process is Markov

$$
q(x_0, x_1, \cdots, x_{T}) = q_0(x_0) \prod_{t=1}^T q_{t|t-1}(x_{t}|x_{t-1}).
$$

The reverse process variational ansatz can be similarly constructed

$$
p(x_0,x_1, \cdots, x_{T}) = p_{T}(x_{T}) \prod_{t=1}^T p_{t-1|t}(x_{t-1}|x_{t})
$$

which can be interpreted as a series of consecutive priors for the physical observation $x_0$.

If we marginalize the latent variables $x_{1,\cdots,T}$, we get the objective function or observable likelihood

$$
p(x_0) = \int \mathcal D x_{1,\cdots,T} \; p(x_0,x_1, \cdots, x_{T})\equiv\int_{1,\cdots,T} \, p_{0,1,\cdots,T}.
$$

## MLE and variational approach

We will derive the lower bound of maximum likelihood objective $\mathbb E_{x_0 \sim q_0} \log p(x_0)$ and then show that it is related to a KL-divergence up to a constant.

_Proof._
The original approach would expand the $p$ joint density

$$
p(x_0) = \int_{1,\cdots,T} \; p_{T}(x_{T}) \prod_{t=1}^T p_{t-1|t}(x_{t-1}|x_{t})
$$

assume Markov property of backward process. However, without assumption of Markov property, we can still insert an identity and get

$$
p(x_0) = \int_{1,\cdots,T} \; \frac{p_{0,1,\cdots,T}}{q_{0,1,\cdots,T}}q_{0,1,\cdots,T}.
$$

Note that $q_{0,1,\cdots,T} = q_{1,\cdots,T | 0} q_0$, then we can reinterpret the integral

$$
p(x_0) = q_0\mathbb E_{q_{1,\cdots,T|0}} \frac{p_{0,1,\cdots,T}}{q_{0,1,\cdots,T}}.
$$

The log-likelihood average over all data distribution is

$$
\mathbb E_{q_0} \log p(x_0) = \mathbb E_{q_0} \log q_0 +\mathbb E_{q_0}  \log  \mathbb E_{q_{1,\cdots,T|0}} \frac{p_{0,1,\cdots,T}}{q_{0,1,\cdots,T}}.
$$

Use the concavity of logarithm function,

$$
\mathbb E_{q_0} \log p(x_0) \ge \mathbb E_{q_0} \log q_0 + \mathbb E_{q_{0,1,\cdots,T}} \log    \frac{p_{0,1,\cdots,T}}{q_{0,1,\cdots,T}} = L_{\rm ELBO}.
$$

where

$$
L_{\rm ELBO} = - H[q_0] - D_{\rm KL} (q_{0,\cdots,T}|p_{0,\cdots,T}).
$$

The entropy $H[q_0]$ depends on data distribution and does not contain model parameters. Thus, maximizing log-likelihood lower bound $L_{\rm ELBO}$ is equivalent to minimizing KL-divergence between forward process joint density and backward process joint density.

> Regardless of Markov property of the processes, max likelihood lower bound is equivalent to min KL-divergence between variational ansatz and physical forward process.

## Markov variational Ansatz

If the forward process is Markov, then we have

$$
q_{0,1,\cdots,T} = q_0 \prod_{t=1}^T q_{t|t-1}.
$$

Similarly, if we assume the backward joint density can be expanded as

$$
p_{0,1,\cdots,T} =  \left(\prod_{t=1}^T p_{t-1|t} \right)  p_{T}.
$$

The posterior of physical forward process may be represented as

$$
q_{t-1|t} \propto q_{t|t-1} q_{t-1};
$$

however, without the knowledge of the initial state $x_0$, there could be infinity possibilities. Therefore, we fix the initial state, and get probabilies given the fixed $x_0 \sim q_0$

$$
q_{t-1|t, 0} \propto q_{t|t-1, 0} q_{t-1 | 0},
$$

or the equality for $t>1$

$$
q_{t-1|t, 0} q_{t|0} = q_{t, t-1|0} = q_{t|t-1, 0} q_{t-1 | 0}.
$$

Thus, the forward process joint density has an posterior expansion

$$
q_{0,1,\cdots,T} = q_0 q_{1|0}\prod_{t=2}^T q_{t|t-1,0} =
q_0 q_{1|0}\prod_{t=2}^T q_{t-1|t,0}\frac{q_{t|0}}{q_{t-1|0}}
$$

where the last factor telescopes

$$
q_{0,1,\cdots,T} = q_0 \left( \prod_{t=2}^T q_{t-1|t,0} \right) q_{T|0}.
$$

The ratio of forward and backward density can be expanded in the following fashion

$$
\frac{q_{0,1,\cdots,T}}{p_{0,1,\cdots,T}} =\frac{q_{T|0}\,q_0}{p_{T}\,p_{0|1}}\prod_{t=2}^T\frac{q_{t-1|t,0}}{p_{t-1|t,0}}
$$

whose logarithm reads

$$
\log\frac{q_{0,1,\cdots,T}}{p_{0,1,\cdots,T}} =\log\frac{q_0}{p_{0|1}}  + \sum_{t=2}^T\log\frac{q_{t-1|t,0}}{p_{t-1|t,0}} +\log\frac{q_{T|0}}{p_{T}} .
$$

Using the posterior expansion of $q$, the total KL-divergence

$$
D_{\rm KL} (q_{0,\cdots,T}|p_{0,\cdots,T}) \equiv  \sum_{t=1}^T D_{t-1}
$$

where

* $D_0 = D_{\rm KL}(q_0|p_{0|1})$,  

* $D_{t-1} = D_{\rm KL}(q_{t-1|t,0}|p_{t-1|t,0})$ for $t=2,\cdots,T$, and

* $D_{T} = D_{\rm KL} (q_{T|0}|p_{T})$.

The last term $D_{T}$ is a constant with **fixed** distribution $p_{T}$. If add back the entropy term $H[q_0]$, the first term becomes the usual likelihood

$$
L_0 = D_0 + H[q_0] = -\log p_{0|1}
$$

and the total loss becomes a typical variational inference loss: the sum of data negative log-likelihood and a series of prior KL-divergence.

> So far we have not assumed any specific distribution yet. The objective is purely based on the assumption of Markov property.

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
{\bf x}_{t-1} = \alpha^{-\frac12} \left( {\bf x}_{t} - \sqrt{\beta_{t}} {\bf z}\right)
$$

but the posterior mean is

$$
\tilde \mu_{t} ({\bf x}_{t}) = \alpha_{t}^{-\frac12 } \left( {\bf x}_{t} - \sqrt{\frac{1 - \alpha_{t}}{1-\bar \alpha_{t}}} \sqrt{\beta_{t}}{\bf z} \right)
$$

where

$$
0 < \sqrt{ \frac{1-\alpha_{t}}{1-\bar \alpha_{t}} } \le 1 .
$$

Thus, the posterior backward process is subtracking less noise except for $t=1$ than what one would naively do.

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
