# Finite Step Unconditional Markov Diffusion Models

## Categorical diffusion

For the case of categorical diffusion models with $K$-categories, we assume 

* the terminal distribution is a maximum entropy distribution, i.e. uniform distribution over the $K$ classes
  
  $$
  q_{T|0} = q_{T} = p_{T} = {\rm Cat}({\bf x}_T; {\bf 1}/K),
  $$

* the forward transition is a categorical distribution
  
  $$
  q_{t|t-1} = {\rm Cat}({\bf x}_{t}; (1-\beta_t){\bf x}_{t-1} + \beta_t {\bf 1}/K)
  $$
  
  where $0<\beta_{t} \le 1$.

It is useful to introduce additional notations:

* $\alpha_{t} \equiv 1 - \beta_{t}$, and
* $\bar \alpha_{t} \equiv \prod_{t=1}^T \alpha_{t}$.

Also note that $\{{\bf x}\}$ are categorical random variables represented as one-hot vectors. The likelihood

$$
{\rm Cat}({\bf x} | {\bf p}) = \prod_{k=1}^K p_k^{x_k} = {\bf x} \cdot {\bf p}
$$

### Markov property

We will show that 

$$
q_{t|0} = {\rm Cat}(\bar\alpha_t {\bf x}_0 + (1-\bar\alpha_t){\bf 1}/K)
$$

which suggests we can sample any ${\bf x}_t$ without the knowledge of ${\bf x}_{1,\cdots, t-1}$.

*Proof*

(i) For $t=1$, following the transition rule

$$
q_{1|0} = {\rm Cat}(\bar \alpha_1 {\bf x}_0 + (1-\bar\alpha_1){\bf 1}/K).
$$

(ii) For $t=2$,

$$
q_{2|0} = \int_1 q_{2, 1 | 0} = \int_1 q_{2|1}q_{1|0}
$$

where

$$
q_{2|1} = {\rm Cat}(\alpha_2 {\bf x}_1 + (1-\alpha_2) {\bf 1}/K) = {\bf x}_2 \cdot (\alpha_2 {\bf x}_1 + (1-\alpha_2) {\bf 1}/K)
$$

with ${\bf x}_1\cdot {\bf x}_2 = \delta_{x_1 x_2}$ and ${\bf x}_2 \cdot {\bf 1} = 1$. 

Then we have

$$
q_{2|1} = \alpha_2 \delta_{x_1 x_2} + (1-\alpha_2)/K,
$$

and similarly

$$
q_{1|0} = \bar \alpha_1 \delta_{x_0 x_1} + (1-\bar \alpha_1) / K.
$$

Thus, 

$$
q_{2|0} = \sum_{x_1} (\alpha_2 \delta_{x_1 x_2} + (1-\alpha_2)/K )(\bar \alpha_1 \delta_{x_0 x_1} + (1-\bar \alpha_1) / K)
$$

which gives

$$
q_{2|0} = \bar \alpha_2 \delta_{x_2 x_0} + (1-\bar\alpha_2)/K
$$

where 

$$
1-\bar\alpha_2 = \alpha_2 (1 - \bar\alpha_1) + \bar\alpha_1 (1-\alpha_2) + (1-\alpha_2)(1-\bar\alpha_1).
$$

Hence, 

$$
q_{2|0} = {\rm Cat}(\bar \alpha_2 {\bf x}_0 + (1-\bar\alpha_2){\bf 1}/K ).
$$

(iii) For $t > 0$, assume we have

$$
q_{t-1|0} = {\rm Cat}(\bar \alpha_{t-1} {\bf x}_0 + (1-\bar\alpha_{t-1}){\bf 1}/K).
$$

Use the transition probability, and marginalize ${\bf x}_{t-1}$,

$$
q_{t|0} 
= \int_{t-1} q_{t|t-1}q_{t-1|0} \\
= \int_{t-1} (\alpha_t \delta_{x_t x_{t-1}} + (1-\alpha_t)/K)
(\bar\alpha_{t-1} \delta_{x_{t-1} x_{0}} + (1-\bar\alpha_{t-1})/K) \\
= \bar \alpha_t \delta_{x_t x_0} + (1 - \bar \alpha_t) / K.
$$

Therefore, we have shown that

$$
q_{t|0} = {\rm Cat}(\bar \alpha_t {\bf x}_0 + (1 - \bar\alpha_t) {\bf 1}/K).
$$

Note that it is important that the terminal distribution is a uniform distribution, otherwise the Markov property does not hold.

### Posterior is categorical

We have physical reverse process

$$
q_{t-1|t, 0} \propto q_{t|t-1} q_{t-1 | 0}
$$

where the RHS is a vector

$$
(\alpha_t \delta_{x_t x_{t-1}} + (1-\alpha_t)/K)
(\bar\alpha_{t-1} \delta_{x_{t-1} x_{0}} + (1-\bar\alpha_{t-1})/K)
$$

with $x_{t-1} = 1, \cdots, K$.

Thus,

$$
q_{t-1|t,0} = {\rm Cat}({\bf x}_{t-1}; {\bf f}({\bf x}_t, {\bf x}_0))
$$

where

$$
\tilde f_k({\bf x}_t, {\bf x}_0) = (\alpha_t \delta_{x_t k} + (1-\alpha_t)/K)
(\bar\alpha_{t-1} \delta_{x_{0}k} + (1-\bar\alpha_{t-1})/K).
$$

and

$$
f_k({\bf x}_t, {\bf x}_0) = \tilde f_k({\bf x}_t, {\bf x}_0) / \sum_{k=1}^K
 \tilde f_k({\bf x}_t, {\bf x}_0).
$$

#### Approximate log-posterior

The posterior may be rewritten in one-hot format as

$$
\vec q_{t-1|t,0} \propto \left ( \alpha_t \vec x_t + \frac{1}{K}(1 - \alpha_t) \vec 1  \right )
\odot \left ( \bar \alpha_{t-1} \vec x_0 + \frac{1}{K}(1 - \bar \alpha_{t-1}) \vec 1  \right ) \equiv \vec f
$$

up to a normalization $\vec 1 \cdot \vec f$. If one take the logarithm of $\vec f$,

$$
\log \vec f = \log \left ( \alpha_t \vec x_t + \frac{1}{K}(1 - \alpha_t) \vec 1  \right )
+ \log \left ( \bar \alpha_{t-1} \vec x_0 + \frac{1}{K}(1 - \bar \alpha_{t-1}) \vec 1  \right )
$$

where the first term does not need to be differentiable, while the second term

$$
{\log(\cdots)} \ge 
\bar \alpha_{t-1} \log \vec x_0 
+ (1 - \bar \alpha_{t-1}) \log \frac{1}{K} 
$$

Also note that

$$
\frac{\vec f}{\vec 1\cdot \vec f} 
= \frac{\vec f_{-}+\delta \vec f}{\vec 1 \cdot (\vec f_{-}+\delta \vec f)} 
= \frac{\vec f_{-}}{\vec 1 \cdot \vec f_{-}} + \frac{\delta \vec f}{\vec 1 \cdot \vec f_{-}}- \mathcal O (\delta \vec f \odot \delta \vec f)
$$

Therefore, 

$$
\log \frac{\vec f}{\vec 1 \cdot \vec f} \ge \log \frac{\vec f_{-}}{\vec 1\cdot \vec f_{-}}
$$



### Variational Ansatz

We choose a category distribution as the variational Ansatz

$$
p_{t-1|t} = {\rm Cat}({\bf x}_{t-1}; {\bf f}({\bf x}_t, \hat {\bf x}_0))
$$

where $\hat {\bf x}_0 = {\bf g}({\bf x}_t, t)$ is the neural network model.

The KL-divergence between physical and variational reverse process is

$$
D_{t-1} = D_{\rm KL}(q_{t-1|t,0}|p_{t-1|t}) = \sum_k q_k \log \frac{q_k}{p_k}
$$

which is

$$
D_{t-1} = - \sum_k f_k ({\bf x}_t, {\bf x}_0) \log f_k({\bf x}_t, {\bf g}({\bf x}_t, t)) + {\rm const.}
$$

#### Training and inference algorithms

##### Training

* Sample
  * grab a sample ${\bf x}_0 \sim q_0$ 
  * pick a time $t \sim {\sf Uniform}(1,\cdots,T)$
  * sample an ${\bf x}_t$
* Feed ${\bf x}_{t}, t$ to model $\bf g$
* Compute ${\bf f} ({\bf x}_t, {\bf x}_0)$ and ${\bf f} ({\bf x}_t, \hat {\bf x}_0)$
* Minimize $D_{t-1}$

##### Inference

* Sample $x_{T} \sim {\rm Cat}({\bf 1}/K)$
* Loop $t = T,\cdots, 1$
  * Compute $\hat {\bf x}_0 = {\bf g}({\bf x}_t, t))$
  * Sample ${\bf x}_{t-1} \sim {\rm Cat}({\bf f}({\bf x}_t, \hat {\bf x}_0))$
* Return ${\bf x}_0$

---

# References

[[1503.03585] Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585)

[[2102.05379] Argmax Flows and Multinomial Diffusion: Learning Categorical Distributions](https://arxiv.org/abs/2102.05379)
