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

# References

[[1503.03585] Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585)

[[2006.11239] Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

[[2102.05379] Argmax Flows and Multinomial Diffusion: Learning Categorical Distributions](https://arxiv.org/abs/2102.05379)
