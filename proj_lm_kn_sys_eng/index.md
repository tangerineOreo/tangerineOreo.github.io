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

**Token**  
&emsp;&emsp;Tokenizer 123...1000... Length information but no dimensional attributes  
&emsp;&emsp;&emsp;&emsp;Dense information, inability to handle polysemy, multi-word combinations  
&emsp;&emsp;One-hot (1,0,0) Dimensional attributes but no length information  
&emsp;&emsp;&emsp;&emsp;Semantic space, all orthogonal, sparse  
Dimensionality reduction/compression facilitates 1D dimensionality increase/reconstruction of compressed data  
Tokenizer first, then one-hot, followed by semantic space

**Matrix** multiplication/Vector-matrix operations/Matrix effects determined by matrix/Spatial transformation/Spatial relationships/New coordinate system transformation/Linear point-to-point mapping/Dimensionality increase, decrease, rotation, scaling/Invariant inter-vector relationships

**Neural network**/Non-linear spatial transformations/Dimension-increasing kernel functions/Hierarchical feature extraction with dimensionality changes  
&emsp;&emsp;Matrix operations/Dimensionality changes, parameters, as above  
&emsp;&emsp;Activation functions/Non-linear/Non-one-to-one correspondence  
Matrix operations/GPU

&emsp;&emsp;Fully connected networks cannot handle polysemy/Tensors parallel to network  
&emsp;&emsp;Context required, leading to attention mechanisms

**Attention**<br>
1960s non-parametric attention/Distance kernel regression  
&emsp;&emsp;Data (xi,yi) or vector y, query x  
&emsp;&emsp;Scalar form f(x)=sum_i alpha(x,xi)*yi, attention weights alpha  
Parameterized attention/Introduction of learnable parameters  
Score/Similarity/Correlation of q·k, Weights/Softmax of scores  
&emsp;&emsp;Score function design Wqk custom

**Structure**  
Encoder non-autoregressive, Decoder autoregressive, KV cross Q/Translation/decoder-only removed  
Input encoding module/Embedding + positional semantics, Feature module, Task requirement module/Linear + Softmax  
Attention, Fully connected feedforward 1ReLU2/+ResNet+Norm  
&emsp;&emsp;Attention position corresponds to V sequence semantic space influence  
&emsp;&emsp;W matrix din=dout, Q matrix K matrix/sqrt(dout)/Vectors in QK are dout-dimensional independent N(0,1)/Product N(0,dout)  
&emsp;&emsp;Din dimension split heads/Dout split heads, Feedforward dimensionality changes, Hidden layer d-task linear layer logits

Feature of training inference<br>
Training requires a mask, because attention carries historical information, and because we want teacher forcing with parallel loss computation, rather than feeding the sequences in one at a time and taking the last row (that works but is inefficient; batching successive prefixes one after another is also inefficient, worse than parallelism within a single sequence)<br>
&emsp;&emsp;Each row is only affected by the history<br>
Inference: batch, seq, dim/vocab; take the last row; custom random generation<br>
&emsp;&emsp;Prefill: process the prompt sequence / there is no KV cache yet and the KV cache is built; during generation, input one token batch, 1, not a sequence<br>
Inference during training: parameters remain unchanged

### Machine learning and deep learning

**Principle**<br>
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
Optimization constraint equivalence, L1 sparse solution/extreme values more likely on coordinate axes, L2 weight decay/during gradient update<br>
Occam's razor/no L3 L4 Ln/Taylor expansion primary and secondary, no free lunch theorem<br>
Monte Carlo<br>
&emsp;&emsp;E[f(x)] = Integral(p\*f\*dx) = Discrete average f(xi) = Sum of probability * function<br>
CNN local, RNN memory

Linear model regression / Derivative = 0 yields an explicit solution<br>
&emsp;&emsp;Perceptron / sign activation function +-1 / Batch size 1 update<br>
Logistic sigmoid(x-y) = ex/(ex+ey)<br>
Softmax and noise contrastive estimation, infoNCE

KL directly measures the difference between distributions<br>
When determining labels or distributions, cross-entropy is equivalent to the KL

&emsp;&emsp;Information **Entropy**<br>
&emsp;&emsp;&emsp;&emsp;Quantification of information; definition f(xy)=f(x)+f(y), i.e., -log2(x), where 2 represents bits<br>
&emsp;&emsp;&emsp;&emsp;The degree of transition from uncertainty to certainty<br>
&emsp;&emsp;&emsp;&emsp;Information content or information entropy contributed to the system = cumulative probability proportion x, i.e., expectation<br>
&emsp;&emsp;Relative Entropy / KL Divergence<br>
&emsp;&emsp;&emsp;&emsp;Cumulative pi\*log(pi/qi) / Cross-entropy minus sampling baseline information entropy<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Gibbs inequality / Relative entropy KL > 0, -KLpq != KLqp<br>
&emsp;&emsp;&emsp;&emsp;Cross-entropy Loss accumulates over quantity / Baseline is the sample, accumulating multiple probability labels

**Data structure**<br>
[...[...[1,2,3],[]...]...] Vectors compose tensors<br>
Ax = lambda x: Matrix space transformation and eigenvalues; direction unchanged / eigenvectors<br>
Derivatives of Scalar/Vector(m,1)/Matrix(m,l) with respect to Scalar/Vector(n,1)/Matrix(n,k): (m,n), (m,k,n), (m,l,n), (m,l,k,n)<br>
Forward formula computes and stores values; backward partial derivatives are used for computation<br>
&emsp;&emsp;Automatic differentiation: Differentiable/Explicit derivation vs. Numerical differentiation<br>
&emsp;&emsp;Chain rule / With respect to the finally updated parameters<br>
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

**Reinforcement Learning**/agent policy action a, environment state reward<br>
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
Bellman Equation QQ/QV/VV

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

15T tokens of data<br>
Data quality, prompts, and scale determine performance; model engineering / acceleration / cost reduction

Pre-training data cleaning: heuristic filtering / lightweight rules / unsafe / too short / high n-gram repetition within sentences; deduplication; classifier-based rejection sampling to judge high-quality data<br>
Data mixture / general 50%, math and reasoning 25%, code 17%

8B / 4096 dimensions / 32 layers; 70B / 8192 dimensions / 80 layers; vocabulary of 120k<br>
Learning rate: linear warmup followed by cosine annealing<br>
Benchmarks: general / MMLU, code, math, reasoning, tool use, long context, multilingual

Pre-norm, SwiGLU, RoPE, GQA (next section)

### Model trends

Post-norm: norm is applied after x + f(x)<br>
&emsp;&emsp;The norm changes at every pass; slower convergence / requires warmup and learning-rate scheduling; higher performance ceiling<br>
Pre-norm: x + f(norm(x))<br>
&emsp;&emsp;The residual stream stays clean; fast convergence / insensitive to hyperparameters<br>
RMS

GLU / gated linear unit<br>
&emsp;&emsp;FFN: x - w1 - relu - w2 - output; introduces a projection matrix w3 as a gating sigmoid branch output<br>
&emsp;&emsp;(x - w1 - relu) and (x - w3 - sigmoid) multiplied element-wise - w2 - output<br>
SwiGLU: SiLU = x * sigmoid(x), smooth ReLU

Rotary position embedding (RoPE)<br>
&emsp;&emsp;Not absolute positions / sin-cos positional basis with negative exponents and fractional dimensions; x, y rotated by R(m, θ); applied before attention<br>
&emsp;&emsp;Positional attention depends on the relative position R(m-n) / reduces the positional disturbance to the original attention; positional relations over long sequences; position affects attention but not v / v is not encoded<br>
&emsp;&emsp;e^(iθ): rotation as a coordinate transformation / cos θ + i sin θ (Euler's formula)<br>
&emsp;&emsp;Rotating the coordinates of a d-dimensional vector / degrees of freedom explode; orthogonal 2D rotation / the most minimal form; length × single rotation / no physical meaning

MHA / MQA / GQA<br>
&emsp;&emsp;Multi-query attention / all heads share one KV matrix / less computation / but performance drops<br>
&emsp;&emsp;Grouped-query attention / a compromise

Depth and width: d_model / n_layers ≈ 100, vocab size on the order of 100k, hidden dimension ratio d_ffn = 4 * d_model, mostly 2.6-4<br>
Weight decay; dropout used sparingly

Matrix multiplication > norm > bias; trade-off between attention complexity and computational speedup<br>
Innovative module architectures

No GQA compromise / project the KV into a latent space and reconstruct / MLA<br>
Linear attention: dim\*dim instead of seq\*seq<br>
Sparse attention: a few tokens before and after each token, top-k per row, etc.<br>
Attention residual: ResNet with coefficient 1, learnable-parameter coefficient, weighted coefficient in the attention computation<br>
&emsp;&emsp;h_l = h_(l-1) + f(h_(l-1)) = h_(l-2) + f(h_(l-2)) + f(h_(l-1)) = h_0 + f_1 + ... + f_(l-1)<br>
&emsp;&emsp;Residual term h_(l-1)<br>
&emsp;&emsp;Attention residual: before layer l, or the preceding l-1 layers<br>
&emsp;&emsp;k_i and v_i are the outputs of each layer; q is a learnable vector parameter of layer l

Multi-token prediction<br>
&emsp;&emsp;The last hidden layer of the main model connects to a one-layer attention module of the next model, whose hidden layer connects to the one after it, and so on<br>
&emsp;&emsp;Each model outputs a token; the loss of the main model plus the losses of the multiple subsequent models<br>
&emsp;&emsp;Trains the main model's ability to predict multiple future tokens<br>
&emsp;&emsp;Speculative decoding: inference with the multiple subsequent models

### Code notice

## Traning

### Pre-training/CPT

Pre-training from 0, or pick a base model and continue pre-training (CPT); dense models; MoE models / too large<br>
Data acquisition / cleaning / pipeline; pre-training 1-15T tokens, CPT tens of billions of tokens / fine-tuning counted in numbers of samples; pre-training loss accumulated over both input and output; full training; total steps = tokens / (batch * seq_len); one step processes multiple batches; pre-training lr > CPT lr<br>
Pre-training / fine-tuning / RL, evaluation and testing

**Compute**<br>
What billions of parameters use what tens of GPUs; distributed parallel training; tens of days to weeks or months<br>
&emsp;&emsp;GPT-3: 175B parameters, 8000 GPUs, 50 days<br>
C (FLOPs) = 6 * model parameter count P * token count D; time T in seconds = C / number of GPUs / compute per GPU / utilization 0.3-0.5<br>
&emsp;&emsp;K/M/GFLOPS: 10e3 / 10e6 / 10e9; TFLOPS: 10e12, RTX 4090; PFLOPS: 10e15<br>
**Scaling law**: parameter count^-a + pre-training token count^-b<br>
&emsp;&emsp;Compute-optimal ratio 1:20<br>
&emsp;&emsp;In practice possibly over-trained at 1:200: 70B model with 15T tokens, 671B model with 15T tokens<br>
&emsp;&emsp;The inverse ratio / high inference cost

Tokenizer training / word segmentation<br>
&emsp;&emsp;One token per digit; token vocabulary size; leave extra room in the vocabulary and embedding layer; Chinese/English ≈ 2 / 1 word per token<br>
&emsp;&emsp;Vocabulary expansion: multilingual, domain-specific terms / proper nouns; single-word vs. multi-word / encoding-decoding efficiency / context length

**Data pipeline**<br>
Format: documents joined with eos into chunks; one batch contains multiple chunks; fixed length, physically contiguous but logically not contiguous<br>
Pre-training data acquisition / on the order of several T<br>
&emsp;&emsp;Open-source datasets; crawlers / MediaCrawler; document / text parsing / MinerU<br>
Data cleaning<br>
&emsp;&emsp;Manual cleaning yields the initial high-quality data; the high- and low-quality data are used to train a fastText classifier / similar to BERT / scores in the 0-1 range, rejection sampling<br>
&emsp;&emsp;Deduplication and recall<br>
&emsp;&emsp;n-gram shingling into sets + MinHash + LSH, balancing recall, precision, and speed<br>
&emsp;&emsp;Term frequency (TF): counts, any model / bag-of-words (BoW): TF forms a document vector; TF * inverse document frequency (IDF) / the more common a word, the smaller its IDF / tf-idf: the uniqueness of a word in this document relative to other or all documents<br>
&emsp;&emsp;Embedding space: BERT encodes the sequence with the history of every word; take the first CLS; average pooling; max pooling; vector similarity or clustering; cross-attention matrix between two sequences - produces a score; concatenate two sentences, attention, CLS - linear layer - score<br>
&emsp;&emsp;Data units: merge or compare for overlap; deduplication between documents / paragraphs / sentences, and between the training and test sets<br>
&emsp;&emsp;Filtering: URL filtering, regex filtering / based on words, document-level filtering, sentence-level filtering<br>
&emsp;&emsp;Everyday work-style sentences / mostly uppercase / purely numeric / symbols and tags / specific words such as like, follow, share<br>
&emsp;&emsp;Iterate: keep looking for more data to clean

Data mixture: BERT classification; Chinese / English / code ratio 4:4:2; keep the mixture when continuing training / prevent catastrophic forgetting / scenario ratio 0.15<br>
wandb, TensorBoard<br>
OpenCompass: AIME math competition, Codeforces, MATH-500 math problems, MMLU general, SWE-bench Verified software engineering

Datasets Hugging Face

### MoE

MoE / the outputs of multiple modules are summed by proportion; the gate network model outputs the proportions<br>
&emsp;&emsp;nn.ModuleList (list []), shape operations and weighted computation<br>
Sparse MoE / Switch Transformer / the switch selects an FFN<br>
Shared-expert sparse MoE / DeepSeek; torch.topk over the experts other than the shared one<br>
MoE activation 0.05 / 0.1 / 0.2

### SFT/instruct fine-tuning

Scenarios for fine-tuning<br>
&emsp;&emsp;1. response style; 2. QA pairs; 3. domain knowledge; 4. code / math ability; 5. function call; 6. module enhancement in agents / workflows<br>
Differences from pre-training<br>
&emsp;&emsp;QA data format, chat template<br>
&emsp;&emsp;Loss computed on the answer only, zeroed out on the question<br>
&emsp;&emsp;Data augmentation / adding noise / NEFTune: noise in the input space taken as 0-10 / sqrt(dim), uniform distribution U

**PEFT**<br>
Prompt tuning / parameters frozen / each task has its own fixed set of input prompt tokens<br>
&emsp;&emsp;Hard prompt / actual language; soft prompt / vectors in the embedding space / are themselves parameters<br>
&emsp;&emsp;P-tuning / trains a representation network / soft prompts as prefix, in the middle, or at the end<br>
&emsp;&emsp;Prefix tuning / added only to the KV matrices of every layer<br>
Add an adapter module / ResNet-style / nonlinear dimensionality reduction then linear restoration; task adaptation / pluggable<br>
LoRA / original matrix + the product of a decomposed fine-tuning update / rank r = 4 or 16 / mrrn; task adaptation / pluggable<br>
&emsp;&emsp;Matrix A initialized with random Gaussian N(0, sigma); matrix B initialized to 0<br>
&emsp;&emsp;Both A and B Gaussian-initialized / random noise added on top of the pre-trained model / unstable<br>
&emsp;&emsp;Both A and B zero-initialized / chain rule dL/dy * dy/dw * dw/dA or dB gives parameter updates of 0<br>
&emsp;&emsp;A reduces the dimension, B increases it / A must be non-zero<br>
Works well in the FFN<br>
QLoRA / NF4<br>
reLoRA: multiple restarts while continuing pre-training

**Overfitting, data volume, model training**<br>
Train for 1 epoch / with a lot of data there is no need for multiple passes; with little data / fewer than 3 epochs<br>
&emsp;&emsp;Little data: easy to overfit<br>
&emsp;&emsp;Many epochs: easy to overfit<br>
Full fine-tuning is easy to overfit and needs more data; PEFT is equivalent to strong regularization<br>
&emsp;&emsp;Full fine-tuning: 100k samples; PEFT: 5k / 10k<br>
Scenarios 3-6: full fine-tuning preferred / PEFT loses performance<br>
Fine-tuning a -base model requires more data and is harder than an -instruct model

Total steps = number of samples / batch * epochs; lr scheduling<br>
&emsp;&emsp;Early stopping on the validation set; patience: the number of epochs or steps to keep training while the loss is not decreasing; generally not needed for SFT

**Industry consensus**<br>
&emsp;&emsp;The quality and diversity of prompts / QA pairs matter far more than the volume of data<br>
&emsp;&emsp;LIMA: Less Is More for Alignment — the idea of constructing data; résumé<br>
&emsp;&emsp;Synthetic data is important / automatically generated; synthesize in different ways to reduce concentrated bias<br>
&emsp;&emsp;Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone — résumé<br>
&emsp;&emsp;Don't add pre-training data; add general instructions as regularization to prevent overfitting and catastrophic forgetting<br>
&emsp;&emsp;If SFT injects too much knowledge / it should be continued pre-training and RAG instead; alignment tax; generalization and diversity; decline in the quality of expression in answers

Fine-tuning cost: prompts, RAG, etc. come first<br>
Outside enterprises it is generally not used locally: API, API + RAG, small model + local RAG, fine-tuning of small-to-medium models

Data synthesis<br>
&emsp;&emsp;Start from an existing split; tag every data item with a task type<br>
&emsp;&emsp;Random sampling with a seed; stratified sampling<br>
&emsp;&emsp;Prompt other large models to rewrite, expand, and generalize the questions and generate the answers / distillation; GPT for English, Qwen-72B / DeepSeek-MoE for Chinese<br>
&emsp;&emsp;Provide CoT wherever possible / step-by-step reasoning and analysis<br>
&emsp;&emsp;Answers in different formats and styles, e.g. markdown, JSON, and the writing style of the question itself<br>
&emsp;&emsp;In industry: train a reward model or use rule-based rejection sampling to select the data for fine-tuning and RL<br>
Multi-source data, multiple domains and personas, multiple QA styles + automatic augmentation and distillation-based generation + rule-based or model-based filtering

**GPU memory calculation**

1 KB = 2^10 = 1024 B/ytes; 1 B/yte = 8 bits; the precision of a number: FP32 / FP16, i.e. 32 / 16 bits<br>
Data volume: 1B = 1000^3; FP32 / FP16: 4GB / 2GB<br>
GPU memory at inference<br>
&emsp;&emsp;Model weights + KV + temporary activations<br>
&emsp;&emsp;Quantization for loading and for computation; KV cache with PagedAttention; distributed multi-GPU parallelism; FlashAttention; concurrency optimization<br>
GPU memory during training<br>
&emsp;&emsp;Mixed precision: 16-bit weights for the forward and backward computation + temporary activations + loss + temporary 16-bit gradients + 32-bit optimizer computation for the update<br>
&emsp;&emsp;Distributed multi-GPU parallelism; LoRA; NF4-quantized weights (QLoRA, slower); FlashAttention

Optimizer computation / gradient-descent parameter updates; the values are small / FP32<br>
&emsp;&emsp;The updated weights, or m*n per layer / 1; the corresponding gradients / 1; optimizer state m first- & second-order coefficients 0.9 / 0.999 / 1 or 2 or k<br>
Forward activations: batch * seq_len * dim * layers * (C_attn + C_mlp + 0 with FlashAttention) + batch * seq_len * vocab; not only the output of each layer — the structural and intermediate quantities of the computation process are also needed / estimated at 10+; the vocab input of the first (input) layer does not need to be considered / its output dim matches the following layers and is already accounted for<br>
KV cache: 2 (K and V) * layers * dim * max sequence length * batch

A100 / H100 80GB<br>
After LoRA: the number of model parameters is unchanged, activations are unchanged, only the added LoRA matrices are trained, gradients are 1/100, optimizer computation is reduced, and the base model can directly add or subtract the LoRA matrix values<br>
Memory estimation: minimal fine-tuning / parameter count in B * 2GB+; LoRA / * 3GB+; full fine-tuning / * (2 + a + 2 + 4 + 8 + 4)GB+

**Trainning parameters**<br>
Learning rate: full fine-tuning 1-5e-5, PEFT 1-3e-4, pre-training 1e-4 - 1e-3<br>
Per-GPU batch size 2-8 / limited by memory; gradient accumulation 1-8 / simulates a larger batch; effective batch = number of GPUs * accumulation<br>
L2 weight decay 0-0.01

**Implementation** of fine-tuning<br>
LlamaFactory<br>
Unsloth

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











