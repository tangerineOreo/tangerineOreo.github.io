---

title: Project

---


<script>
  const h1 = document.querySelector('h1');
  if (h1) {
    h1.remove();
  }
</script>


System, understanding and code quick check of large model training, inference, applications and projects

## Catalog
**Basis**
- Origin & transformer
- Machine learning and deep learning
- Language model
- Large model pre
- Model trends

**Traning**
- Pre-training/CPT
- MoE
- SFT/instruct fine-tuning
- Distillation
- Reinforcement learning

**Training & Inference Optimization**
- Data, device and algorithm
- API
- Function call
- Prompt

**Applications**
- Multimodal
- RAG
- Agent
- Engineering and project

## Basis
### Origin & transformer
Token  
&emsp;&emsp;Tokenizer 123...1000... Length information but no dimensional attributes  
&emsp;&emsp;&emsp;&emsp;Dense information, inability to handle polysemy, multi-word combinations  
&emsp;&emsp;One-hot (1,0,0) Dimensional attributes but no length information  
&emsp;&emsp;&emsp;&emsp;Semantic space, all orthogonal, sparse  
Dimensionality reduction/compression facilitates 1D dimensionality increase/reconstruction of compressed data  
Tokenizer first, then one-hot, followed by semantic space

Matrix multiplication/Vector-matrix operations/Matrix effects determined by matrix/Spatial transformation/Spatial relationships/New coordinate system transformation/Linear point-to-point mapping/Dimensionality increase, decrease, rotation, scaling/Invariant inter-vector relationships

Neural networks/Non-linear spatial transformations/Dimension-increasing kernel functions/Hierarchical feature extraction with dimensionality changes  
&emsp;&emsp;Matrix operations/Dimensionality changes, parameters, as above  
&emsp;&emsp;Activation functions/Non-linear/Non-one-to-one correspondence  
Matrix operations/GPU

&emsp;&emsp;Fully connected networks cannot handle polysemy/Tensors parallel to network  
&emsp;&emsp;Context required, leading to attention mechanisms

1960s non-parametric attention/Distance kernel regression  
&emsp;&emsp;Data (xi,yi) or vector y, query x  
&emsp;&emsp;Scalar form f(x)=sum_i alpha(x,xi)*yi, attention weights alpha  
Parameterized attention/Introduction of learnable parameters  
Score/Similarity/Correlation of q·k, Weights/Softmax of scores  
&emsp;&emsp;Score function design Wqk custom

Structure  
Encoder non-autoregressive, Decoder autoregressive, KV cross Q/Translation/decoder-only removed  
Input encoding module/Embedding + positional semantics, Feature module, Task requirement module/Linear + Softmax  
Attention, Fully connected feedforward 1ReLU2/+ResNet+Norm  
Content  
&emsp;&emsp;Attention position corresponds to V sequence semantic space influence  
&emsp;&emsp;W matrix din=dout, Q matrix K matrix/sqrt(dout)/Vectors in QK are dout-dimensional independent N(0,1)/Product N(0,dout)  
&emsp;&emsp;Din dimension split heads/Dout split heads, Feedforward dimensionality changes, Hidden layer d-task linear layer logits  

### Machine learning and deep learning
**Principle Data, Model, Objective Function**<br>
Training set/model parameters, validation set/model hyperparameters/comparative model selection<br>
Training error generalization error, model bias variance data noise, overfitting underfitting, model complexity data complexity regularization<br>
Loss/objective function + regularization term/parameter term<br>
&emsp;&emsp;Least squares/MSE, likelihood/MLE/probability distribution, cross-entropy, other objectives<br>
MLE log = summed cross-entropy, under Gaussian noise assumption MLE is equivalent to summed MSE<br>
&emsp;&emsp;sigmoid/simplest softmax/single class<br>
&emsp;&emsp;&emsp;&emsp;Neural network equivalent to MLE/neural network likelihood log/summed cross-entropy MSE<br>
Bayesian/maximum a posteriori/likelihood\*prior/log likelihood + log prior/regularization<br>
softmax/maximum entropy/e/log, summed entropy/multiple probabilities multiple classes<br>
&emsp;&emsp;Maximum entropy for unknown system information/maximum entropy of system based on known information<br>
&emsp;&emsp;When solving for a distribution or given Z finding a distribution, the linear case of maximum entropy solution<br>
&emsp;&emsp;&emsp;&emsp;Design and construct random variables/distributions/expectations for deterministic data<br>
&emsp;&emsp;&emsp;&emsp;Lagrange multiplier/entropy + constraints<br>
&emsp;&emsp;Maximum entropy determines the model, optimization determines parameters/Lagrange multiplier = MLE<br>
Maximum likelihood/real-world dataset corresponding to theoretical distribution parameters, theoretical distribution/prior/unknown known/custom constraints<br>
&emsp;&emsp;Objective: likelihood + prior<br>
Probability corresponds to event occurrence, random variable takes self-determined values - event occurrence<br>
From softmax to defining Noise Contrastive Estimation (NCE) function<br>
Optimization constraint equivalence, L1 sparse solution/extreme values more likely on coordinate axes, L2 weight decay/during gradient update<br>
Occam's razor/no L3 L4 Ln/Taylor expansion primary and secondary, no free lunch theorem<br>
CNN local, RNN memory<br>



&emsp;&emsp;Information **Entropy**<br>
&emsp;&emsp;&emsp;&emsp;Quantification of information; definition f(xy)=f(x)+f(y), i.e., -log2(x), where 2 represents bits<br>
&emsp;&emsp;&emsp;&emsp;The degree of transition from uncertainty to certainty<br>
&emsp;&emsp;&emsp;&emsp;Information content or information entropy contributed to the system = cumulative probability proportion x, i.e., expectation<br>
&emsp;&emsp;Relative Entropy / KL Divergence<br>
&emsp;&emsp;&emsp;&emsp;Cumulative pi\*log(pi/qi) / Cross-entropy minus sampling baseline information entropy<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Gibbs' inequality / Relative entropy KL > 0, -KLpq != KLqp<br>
&emsp;&emsp;&emsp;&emsp;Cross-entropy Loss accumulates over quantity / Baseline is the sample, accumulating multiple probability labels

[...[...[1,2,3],[]...]...] Vectors compose tensors<br>
Ax = lambda x: Matrix space transformation and eigenvalues; direction unchanged / eigenvectors<br>
Derivatives of Scalar/Vector(m,1)/Matrix(m,l) with respect to Scalar/Vector(n,1)/Matrix(n,k): (m,n), (m,k,n), (m,l,n), (m,l,k,n)<br>
Forward formula computes and stores values; backward partial derivatives are used for computation<br>
&emsp;&emsp;Automatic differentiation: Differentiable/Explicit derivation vs. Numerical differentiation<br>
&emsp;&emsp;Chain rule / With respect to the finally updated parameters<br>
Linear model regression / Derivative = 0 yields an explicit solution<br>
&emsp;&emsp;Perceptron / sign activation function +-1 / Batch size 1 update<br>
Monte Carlo<br>
&emsp;&emsp;E[f(x)] = Integral(p\*f\*dx) = Discrete average f(xi) = Sum of probability * function<br>
Logistic sigmoid(x-y) = ex/(ex+ey)<br>
Bias b is a one-dimensional vector

**Weight Decay**<br>
&emsp;&emsp;lambda=0/no effect, lambda->inf/parameters->0<br>
&emsp;&emsp;The stretching balance between gradients of loss and regularization term on parameters<br>
**Dropout**<br>
&emsp;&emsp;Add noise between layers/outputs of fully connected layers in hidden layers<br>
&emsp;&emsp;Hope E[xi1]=xi<br>
&emsp;&emsp;&emsp;&emsp;xi1=0 with probability p, xi1=xi/(1-p) with probability 1-p<br>
**Numerical Stability**<br>
&emsp;&emsp;Gradient product/explosion/vanishing<br>
&emsp;&emsp;&emsp;&emsp;ResNet, clipping, normalization mean 0 variance 1 Gaussian N(0,1)<br>
&emsp;&emsp;&emsp;&emsp;ReLU is better than tanh and sigmoid<br>
Regularization is used for training, balancing model and data/fitting/generalization

**Norm** stabilizes training and accelerates convergence, linear function can learn parameters/with bias<br>
&emsp;&emsp;BatchNorm/within mini-batch/inconsistent between training and inference/vision<br>
&emsp;&emsp;LayerNorm/independent of other samples/token and feature dimensions/N(0,1)/consistent between training and inference/language/autoregressive/different sentence lengths vary<br>
&emsp;&emsp;&emsp;&emsp;RMSnorm/root mean square/scale coefficient parameter is learnable, reduces computation and accelerates

**Optimization Algorithms**<br>
&emsp;&emsp;Gradient/sample<br>
&emsp;&emsp;&emsp;&emsp;1st-order, 0th-order/approximate gradient/derivative defined as expectation over small distances in all directions<br>
&emsp;&emsp;&emsp;&emsp;Batch gradient/full, stochastic gradient/random one, mini-batch stochastic gradient/random m<br>
&emsp;&emsp;1st-order m/resistance/1 weight m and gradient<br>
&emsp;&emsp;AdamW adaptive m, m/sqrt(2nd-order m)/scaling decay, 2nd-order m/corresponding gradient squared<br>
&emsp;&emsp;&emsp;&emsp;Wide parameter interval is insensitive to learning rate<br>
&emsp;&emsp;&emsp;&emsp;Direction and step size, stable and fast, automatic speed adjustment

**Reinforcement Learning Framework**/agent policy action a, environment state reward<br>
&emsp;&emsp;Experience sarsa updating network, independent, current s policy action<br>
&emsp;&emsp;On-policy/off-policy/training policy vs selection time/behavior vs target policy/arbitrary historical randomness<br>
&emsp;&emsp;&emsp;&emsp;Unrelated to offline/online/concept is vague, distinction between policy historical data and historical policy<br>
Q-learning-DQN/q-policy, SARSA-neural network sarsa/q<br>
Reinforce-policy network/v-value/q and policy/full sampling q, AC/full network<br>
&emsp;&emsp;A2C/A3C, DDPG/TD3/SAC, TRPO/PPO<br>
Parameter Soft Update<br>
r(s,a chain) expansion, randomness a sampling s transition, q/s,a and v/s future long-term accumulation<br>
v=E[q]=integral pi\*q<br>
&emsp;&emsp;Adding or subtracting pi\*b(s) has no effect, derivative of integral pi = derivative of 1 = 0<br>
&emsp;&emsp;Gradient/derivative<br>
&emsp;&emsp;&emsp;&emsp;First approximate then derive, sampling from pi distribution, pi\*q sum is biased estimation, q average is unbiased but no parameter derivative<br>
&emsp;&emsp;&emsp;&emsp;So first derive then approximate, policy gradient logarithm<br>
Bellman Equation QQ/QV/VV<br>


### Language model

### Large model pre

### Model trends

## Traning
### Pre-training/CPT

### MoE

### SFT/instruct fine-tuning

### Distillation

### Reinforcement learning

## Training & Inference Optimization
### Data, device and algorithm

### API

### Function call

### Prompt

## Applications
### Multimodal

### RAG

### Agent

### Engineering and project

