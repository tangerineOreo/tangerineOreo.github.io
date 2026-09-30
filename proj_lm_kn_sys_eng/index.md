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

Training requires a mask, because attention carries historical information, and because we want teacher forcing with parallel loss computation, rather than feeding the sequences in one at a time and taking the last row (that works but is inefficient; batching successive prefixes one after another is also inefficient, worse than parallelism within a single sequence)<br>
&emsp;&emsp;Each row is only affected by the history<br>
Inference: batch, seq, dim/vocab; take the last row; custom random generation<br>
&emsp;&emsp;Prefill: process the prompt sequence / there is no KV cache yet and the KV cache is built; during generation, input one token batch, 1, not a sequence<br>
Inference during training: parameters remain unchanged

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
CNN local, RNN memory

&emsp;&emsp;Information **Entropy**<br>
&emsp;&emsp;&emsp;&emsp;Quantification of information; definition f(xy)=f(x)+f(y), i.e., -log2(x), where 2 represents bits<br>
&emsp;&emsp;&emsp;&emsp;The degree of transition from uncertainty to certainty<br>
&emsp;&emsp;&emsp;&emsp;Information content or information entropy contributed to the system = cumulative probability proportion x, i.e., expectation<br>
&emsp;&emsp;Relative Entropy / KL Divergence<br>
&emsp;&emsp;&emsp;&emsp;Cumulative pi\*log(pi/qi) / Cross-entropy minus sampling baseline information entropy<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Gibbs inequality / Relative entropy KL > 0, -KLpq != KLqp<br>
&emsp;&emsp;&emsp;&emsp;Cross-entropy Loss accumulates over quantity / Baseline is the sample, accumulating multiple probability labels<br>
When determining labels or distributions, cross-entropy is equivalent to the KL<br>
KL divergence directly measures the difference between distributions

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

Probability of a token or a token sequence occurring<br>
Joint probability; the i-th element in the chain-rule factorization of conditional probabilities<br>
Markov assumption; n-gram corpus-count-based probability modeling and prediction / generally n<5, limited length<br>
Methods for obtaining probabilities<br>
&emsp;&emsp;Maximum-probability method<br>
&emsp;&emsp;&emsp;&emsp;Greedy search / 1 candidate / fast / not optimal; exhaustive search / all candidates / impractical; beam search / n candidates × time / 1 path<br>
&emsp;&emsp;Probability-based random sampling / creativity<br>
&emsp;&emsp;&emsp;&emsp;top-k / among the n most probable tokens; top-p / among those whose cumulative probability exceeds x, probabilities normalized before sampling<br>
&emsp;&emsp;&emsp;&emsp;Temperature mechanism / adjusts the level of randomness; candidate probability / T; the numerical gap after normalization<br>
lm-eval, opencompass

&emsp;&emsp;Temperature, greedy search, beam search, top-k, top-p<br>
Creativity, 1-1.5, No, No, 50-100, 0.9-0.95<br>
Accuracy, 0.1-0.5, Yes, 3-5, <=10, <=0.5<br>
Balance, 0.7-1, No, No, 30-50, 0.7-0.9

Similarity and deduplication<br>
&emsp;&emsp;TF, TF-IDF, vectors<br>
<br>
Text clustering, k-means, vector similarity in embedding space<br>
&emsp;&emsp;Dimensionality reduction: PCA / magnitude of variance; LDA (Linear Discriminant Analysis) / projection onto a line; SVD / the V matrix of the data matrix as principal components; UMAP (Uniform Manifold Approximation and Projection) / cross-entropy between high- and low-dimensional representations, low-dimensional visualization<br>
&emsp;&emsp;Singular value decomposition (SVD) / analogous to Fourier and Taylor series<br>
Topic models / category classification / Latent Dirichlet Allocation (LDA), term frequency, bag-of-words, Bayesian inference / BERT / GPT, topic-word distributions / probabilities<br>

### Large model pre
Emergent abilities: in-context learning, CoT, commonsense reasoning / logical reasoning, code, translation, instruction following, etc.<br>
Scaling Law / model performance as a function of model parameter scale and data scale<br>
Transformer-based<br>
&emsp;&emsp;Encoder-only / discriminative tasks / BERT<br>
&emsp;&emsp;Encoder-decoder / comprehensive but costly / T5 prompt<br>
&emsp;&emsp;Decoder-only / autoregressive / generative tasks save on encoding / many emergent tasks / GPT / llama<br>
Training data structure and semantics / sentence formats for different tasks<br>
&emsp;&emsp;Tasks such as reconstructing masked sentences, sentence-meaning matching, sentence-meaning labeling, multi-task, next-token prediction / + prompt labels

Pre-training / unservised and self-supervised; post-training / fine-tuning and RL<br>
Supervised fine-tuning / instruction tuning / QA; DPO / non-RL<br>
&emsp;&emsp;Full fine-tuning vs. reducing parameter count / PEFT / feature modules and parameter reuse

Transfer learning / fine-tuning<br>
&emsp;&emsp;Closer to the optimum / not an initial random point; don't change too much / small space / learning rate / epochs<br>
&emsp;&emsp;Larger and more complex pre-training dataset / better fine-tuning results<br>
&emsp;&emsp;Layers near the input / local features are general / freeze them; layers near the output / global features / learning rate 0 - piecewise, or last layer only<br>
&emsp;&emsp;Lowers cost, speeds up convergence, accuracy not guaranteed

Hallucination<br>
&emsp;&emsp;Data issues: outdated knowledge, boundaries / missing data, bias / incorrect data and prejudice, improper alignment annotation / preferences<br>
&emsp;&emsp;The model itself: long-tail knowledge / low-frequency data, probability sampling / effects of randomness

**BERT** produces a sequence of vectors; the vector of the first CLS token carries the feature information of the whole sequence: CLS vector - fully connected layer - logits - softmax, etc.<br>
Training tasks<br>
&emsp;&emsp;Masked word prediction / MLM, masked language modeling<br>
&emsp;&emsp;Sentence 1 and sentence 2 concatenated; binary classification on CLS for whether it is the next sentence / NSP, next sentence prediction<br>
Fine-tuning tasks, e.g. sentence sentiment classification<br>
When using it, add a task-specific output module and fine-tune

**LLaMA**

### Model trends


Post-norm: norm is applied after x + f(x)<br>
&emsp;&emsp;The norm changes at every pass; slower convergence / requires warmup and learning-rate scheduling; higher performance ceiling<br>
Pre-norm: x + f(norm(x))<br>
&emsp;&emsp;The residual stream stays clean; fast convergence / insensitive to hyperparameters<br>
RMS<br>
<br>
GLU / gated linear unit<br>
&emsp;&emsp;FFN: x - w1 - relu - w2 - output; introduces a projection matrix w3 as a gating sigmoid branch output<br>
&emsp;&emsp;(x - w1 - relu) and (x - w3 - sigmoid) multiplied element-wise - w2 - output<br>
SwiGLU: SiLU = x * sigmoid(x)<br>
<br>
RoPE<br>
<br>
MHA / MQA / GQA<br>
<br>
Depth and width: d_model / n_layers ≈ 100, vocab size on the order of 100k, hidden dimension ratio d_ffn = 4 * d_model, mostly 2.6-4<br>
Weight decay; dropout used sparingly<br>
<br>
Matrix multiplication > norm > bias; trade-off between attention complexity and computational speedup<br>
Innovative module architectures<br>

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

