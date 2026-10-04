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

**Feature of training inference**<br>
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
&emsp;&emsp;TF, TF-IDF, vectors

Text clustering, k-means, vector similarity in embedding space<br>
&emsp;&emsp;Dimensionality reduction: PCA / magnitude of variance; LDA (Linear Discriminant Analysis) / projection onto a line; SVD (Singular Value Decomposition) / the V matrix of the data matrix as principal components; UMAP (Uniform Manifold Approximation and Projection) / cross-entropy between high- and low-dimensional representations, low-dimensional visualization<br>
&emsp;&emsp;&emsp;&emsp;SVD analogous to Fourier and Taylor series

Topic models / category classification / LDA (Latent Dirichlet Allocation), term frequency, bag-of-words, Bayesian inference / BERT / GPT, topic-word distributions / probabilities<br>

### Large model pre

Emergent abilities: in-context learning, CoT, commonsense reasoning / logical reasoning, code, language & translation, instruct following<br>
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
&emsp;&emsp;e^(iθ): rotation as a coordinate transformation / cos θ + i sin θ (Euler formula)<br>
&emsp;&emsp;Rotating the coordinates of a d-dimensional vector / degrees of freedom explode; orthogonal 2D rotation / the most minimal form; length * single rotation / no physical meaning

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

Datasets Hugging Face, modelscope

**Implementation notice**

transformers built on PyTorch nn<br>
&emsp;&emsp;PreTrainedModel: the base class inherits from nn.Module, PretrainedConfig: the base class<br>
&emsp;&emsp;Trainer, TrainingArguments<br>
&emsp;&emsp;AutoProcessor: a smart input processor / text, image, multimodal, audio; AutoModel: a generic model / the task output module is not loaded<br>
&emsp;&emsp;AutoTokenizer loads the tokenizer; AutoModelForCausalLM<br>
&emsp;&emsp;/modeling_outputs: the various output classes / wrap and return hiddens - dim - fully connected - vocab - logits, before normalization into probabilities; loss, attention, kv, etc.<br>
&emsp;&emsp;&emsp;&emsp;Loading a model maps to these automatically; when defining a model you have to return and call them<br>
&emsp;&emsp;&emsp;&emsp;CausalLMOutput / WithPast: the output structure used when the KV cache is enabled<br>
&emsp;&emsp;data_collator<br>
&emsp;&emsp;&emsp;&emsp;default<br>
&emsp;&emsp;&emsp;&emsp;with_padding: dynamic per sample / static global via the tokenizer; the dataset is required to have labels / SFT<br>
&emsp;&emsp;&emsp;&emsp;ForLanguageModeling: mlm=True/False corresponds to BERT / GPT pre-training<br>
The tokenizers library / if the tokenizer is customized<br>
<br>
torch, torch.nn<br>
torch.nn.functional: no learnable parameters / pure computation<br>
dataset and dataloader from torch.utils.data

AutoModel choice / pre-trained model / no task output / base / BertModel<br>
AutoModelFor / pre-trained model / has a task output module but no input module<br>
&emsp;&emsp;CausalLM decoder, Seq2SeqLM encoder-decoder, encoder / with task output / others<br>
<br>
model(**input): input and output are dicts; pass the dict as arguments or pass the dict in; supports object. / dict[]<br>
model.generate: autoregressive calls to model() for logits; the output is an ids sequence tensor, optionally a dict with multiple outputs; batch decode afterwards<br>
&emsp;&emsp;self(input,)<br>
&emsp;&emsp;Takes probabilities and temperature, repetition suppression / penalty<br>
<br>
Loading the tokenizer binds the model and the special text tokens<br>
tokenizer.encode unifies the types across tokenizers; string or list input, dict output / the values are tensors, 1D / 2D<br>
&emsp;&emsp;The tokenizer's output dict supports object. / dict[]<br>
tokenizer.decode / batch_decode: id sequence input / a list or a tensor both work, string or list output, 1D / 2D<br>
The model output includes the input part; the pipeline output for multiple sentences is a list of dicts, [0]['generated_text'] / optionally show only the answer<br>
<br>
Sequence-length padding is only used when batching / aligning dimensions and shapes<br>
&emsp;&emsp;The attention_mask parameter / model(,)<br>
tokenizer padding_side defaults to right / can be changed; GPT / llama recommend left

Trainer training: if the loss is customized, inherit from Trainer and override compute_loss, or return it from model.forward<br>
<br>
The datasets library<br>
&emsp;&emsp;Load datasets from the hub or locally<br>
&emsp;&emsp;&emsp;&emsp;Format: built-in automatic splitting into train and test sets, specifying a split, or when the format has no split, loading it automatically gives 'train'<br>
&emsp;&emsp;After loading, .map(f), where f is the processing function / or returns a processor / tokenizer / feature extractor<br>
datasets.Dataset and DatasetDict match the format but do not inherit from utils<br>
dataset.map(, remove_columns), dataset.remove_columns([''])

The nn.Parameter class / participates in updates / is registered in the model / it is a subclass of torch.Tensor, which does not participate in updates and is not registered<br>
&emsp;&emsp;self.weight / bias / name = nn.Parameter(...) constructs the parameters of one layer<br>
(full path of the parameter name, parameter) tuples / for name, param in model.named_parameters()<br>
&emsp;&emsp;.parameters() gives only the parameters without names / implemented through the .named_parameters() call<br>
A dict of parameter names to parameters / for name in model.state_dict().keys()<br>
<br>
/ nn.init defines how the network parameters are initialized<br>
<br>
trainer.save_model() calls model.save_pretrained to save the weights and the config, which in turn calls torch.save<br>
&emsp;&emsp;torch.save saves only the weights, not config.json: the architecture, including identifiers / network structure / training and inference settings / paths, etc. / can be written by hand<br>
<br>
init self. appears in the state_dict, and can therefore be saved and trained<br>
<br>
Using the same self. layer multiple times means it is shared / the same single memory address / in the actual structure it is still one layer<br>
Assigning by index replaces the layer<br>
Layers are defined through nn.Module's forward<br>
<br>
print(model) prints the structure, including parameter-free layers that are not considered layers from an engineering standpoint<br>
model.named_parameters(): parameters and gradients, freezing layers, per-layer learning rates / grouped parameter lists, parameter counts<br>
&emsp;&emsp;print(name) weight/bias, print(param.shape), param.requires_grad<br>
model.fc1.weight / bias.data / grad; with nn.ModuleList / Sequential it is model[i].weight / bias

### MoE

MoE / the outputs of multiple modules are summed by proportion; the gate network model outputs the proportions<br>
&emsp;&emsp;nn.ModuleList (list []), shape operations and weighted computation<br>
Sparse MoE / Switch Transformer / the switch selects an FFN<br>
Shared-expert sparse MoE / DeepSeek; torch.topk over the experts other than the shared one<br>
MoE activation 0.05 / 0.1 / 0.2

MoE load balancing<br>
&emsp;&emsp;The routing scores are adjusted by a bias b before top-k; b varies with the load; no loss is needed<br>
&emsp;&emsp;&emsp;&emsp;b is adjusted after each micro-batch of tokens<br>
&emsp;&emsp;Loss coefficient: increases automatically when the load is imbalanced, tends to 0 when the load is balanced

MoE token arrangement within a sequence<br>
&emsp;&emsp;For a token of dimension dim, the routing matrix maps dim to the number of experts; the weights are normalized by softmax within the top-k<br>
&emsp;&emsp;The FFNs process the token; a weighted sum over the top-k FFNs<br>
Auxiliary losses can be computed at intermediate layers<br>
&emsp;&emsp;z-loss: a regularization constraint on the logits before the per-layer expert top-k<br>
&emsp;&emsp;&emsp;&emsp;Backpropagation up to that layer; the per-layer losses can be summed; automatic differentiation, with the derivative equal to 0 for unrelated layers<br>
&emsp;&emsp;Load-balancing loss: winner-takes-all — training it further makes the MoE degenerate into a dense model; the fraction of tokens assigned to an expert (actual load) * the probability that the expert is selected by the router; minimizing this expectation is equivalent to load balancing

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

**Implementation of fine-tuning**

AutoDL files in the same region are loaded via the file system onto the instance's temporary disk (tmp) or data disk (data)<br>
Models and data: ModelScope, Hugging Face<br>
unzip, cp, cd

pip install unsloth; unsloth_zoo for model conversion / export / testing and other non-core training functions; bitsandbytes for low-level quantized loading and optimization<br>
&emsp;&emsp;Installing unsloth automatically installs transformers, peft, and accelerate<br>
<br>
&emsp;&emsp;accelerate: a distributed abstraction layer and lightweight orchestrator, built on top of the various underlying libraries<br>
&emsp;&emsp;&emsp;&emsp;Single GPU / data parallelism (DDP) / fully sharded data parallelism (FSDP) — PyTorch, Meta<br>
&emsp;&emsp;&emsp;&emsp;TPU hardware — Google<br>
&emsp;&emsp;&emsp;&emsp;DeepSpeed — Microsoft; ZeRO-3: optimizer states + gradients + model parameters<br>
&emsp;&emsp;&emsp;&emsp;Megatron-LM — NVIDIA; 3D parallelism: data + tensor + pipeline

from unsloth import FastLanguageModel — the integration and support layer; rewritten new features; a library for LM fine-tuning with quantization, acceleration, and distribution<br>
&emsp;&emsp;Load the model and the tokenizer correspondingly, by name / path; optional max_seq_len / dtype / bnb_config with load_in_4bit<br>
&emsp;&emsp;input_text message<br>
&emsp;&emsp;ids_inputs = tokenizer.apply_chat_template<br>
&emsp;&emsp;model(ids_inputs).to(device)<br>
&emsp;&emsp;model.eval(); ids_outputs = model.generate<br>
&emsp;&emsp;tokenizer.decode<br>
pipeline is simpler<br>
Training / inference optimization: .for_training / .for_inference(model)<br>
&emsp;&emsp;still need model.eval / train mode, or trainer.train / evaluate / predict, which runs automatically and returns to train mode

Dataset preparation<br>
The chat template for each model<br>
&emsp;&emsp;String '{}'.format(content): input, cot, output<br>
&emsp;&emsp;&emsp;&emsp;'{name}'.format, f'{name}'<br>
&emsp;&emsp;&emsp;&emsp;+ EOS_TOKEN<br>
&emsp;&emsp;apply_chat_template does it automatically; concatenating cot and output; '{}'.format and f'{}'<br>
The datasets library / format: a list of dictionaries, json<br>
&emsp;&emsp;load_dataset: Hugging Face, local files such as json, loading script .py<br>
&emsp;&emsp;&emsp;&emsp;.train_test_split<br>
&emsp;&emsp;map with a chat-template function or the tokenizer

Model training<br>
FastLanguageModel.for_training(model)<br>
Without PEFT there is no need for .get_peft_model(...) with a LoraConfig<br>
&emsp;&emsp;print(model), target_modules<br>
&emsp;&emsp;lora_alpha / r: W + lora_alpha/r * BA<br>
Trainer<br>
&emsp;&emsp;Dataset: tokenizer to ids; padding is not fixed during training — alignment / dynamic padding via data_collator<br>
&emsp;&emsp;data_collator per batch<br>
&emsp;&emsp;General purpose; generally not used for SFT<br>
TRL's SFTTrainer and TrainingArguments<br>
&emsp;&emsp;The dataset is text; the ids are handled automatically; dataset_text_field='text' is needed because there is more than one field label<br>
&emsp;&emsp;&emsp;&emsp;as SFT requires chat-format markers or the message style<br>
&emsp;&emsp;packing: whether to concatenate up to max_seq_len<br>
&emsp;&emsp;Per-GPU batch / number of samples processed per core; gradient accumulation steps before the update / prevents overfitting / simulates a larger batch — llama 4M<br>
&emsp;&emsp;Learning rate / warmup / scheduling<br>
&emsp;&emsp;Optimizer, quantization, logging, saving, and related settings<br>
<br>
After training<br>
Load with PeftModel or AutoPeftModelFor...; AutoPeftModel does not load the task output module; or load with FastLanguageModel<br>
Merging the model; PEFT is optional<br>

### Distillation

Black-box knowledge distillation: a large model generates data to train a small model<br>
White-box knowledge distillation: the distribution of the large model, KL<br>
&emsp;&emsp;The model distribution is not open-source and is hard to obtain; the models have different output formats

**Implementation of distillation**

mediacrawler<br>
&emsp;&emsp;Enter the project directory and create a virtual environment to run<br>
<br>
Define a function to extract a list of questions from input text with OpenAI<br>
&emsp;&emsp;Large model API involvement, output format setting, and JSON schema<br>
&emsp;&emsp;Determine and filter whether the JSON output is correct<br>
<br>
extract_questions.py<br>
&emsp;&emsp;Read the text field of JSON<br>
&emsp;&emsp;Define a thread pool for multi-threaded processing of the extraction function and input text, with tqdm for progress visualization<br>
&emsp;&emsp;Write results to JSON<br>
<br>
Data deduplication<br>
Vector similarity greater than 0.9, using cosine, Euclidean distance, dot product, TF, TF-IDF<br>
&emsp;&emsp;Client and embedding model calling API, or calling local library functions<br>
&emsp;&emsp;Define a function to calculate similarity<br>
&emsp;&emsp;Define a function to read files and compare each one with existing texts in the list one by one for similarity<br>
<br>
Generate answers<br>
&emsp;&emsp;Load the question file, large model API returns answers, multi-threaded<br>
&emsp;&emsp;Write the question-answer pair dictionary list to JSON

### Reinforcement learning

Fine-tuning has no useless / incorrect / harmful information; RL is needed<br>
Fine-tuning data is limited, and it is imitation; RL scores the generation process, which is generalizable and fuzzy<br>
RL<br>
&emsp;&emsp;Reward model r at each step, value, large model / policy<br>
&emsp;&emsp;Grid scores: the highest achievable future score starting from the current position; train the large model on the path with the highest score

**Sequence reward model**, which does not have to be a neural network / scores based on the outcome and facts<br>
&emsp;&emsp;Data consists of concatenated QA pairs and human preference annotations in pairs<br>
&emsp;&emsp;Training / deep-learning supervised, not RL / contrastive learning on positive and negative data, maximizing the distinction: -log(sigmoid(r_j - r_k))

**Training policy / large model**<br>
&emsp;&emsp;State / autoregressive sequence / the token is updated at each step<br>
&emsp;&emsp;&emsp;&emsp;Input: the question; the large model answers<br>
&emsp;&emsp;r is returned at the last token, when the sequence is complete; there is no r during the intermediate steps

&emsp;&emsp;Proximal Policy Optimization (**PPO**)<br>
&emsp;&emsp;Objective: multi-timestep expectation = policy ratio (new vs. old) in the direction of A - TD via the v network + entropy regularization - relative-entropy KL(RL, SFT), policy ratio<br>
&emsp;&emsp;&emsp;&emsp;Entropy regularization is used in RL / increases exploration and prevents the policy from becoming deterministic<br>
&emsp;&emsp;&emsp;&emsp;L1 / L2 regularization: minimized; entropy regularization: maximized; relative-entropy KL regularization: minimized<br>
&emsp;&emsp;&emsp;&emsp;Policy ratio: the probability of each token of the state sequence under the different policies<br>
&emsp;&emsp;Advantage A = Q - V<br>
&emsp;&emsp;&emsp;&emsp;Adding or subtracting b(s) does not affect it; the Bellman equation for Q and V; subtracting V = E[Q] gives the minimum variance<br>
&emsp;&emsp;On-policy; computing A for a single timestep / Generalized Advantage Estimation (GAE)<br>
&emsp;&emsp;&emsp;&emsp;Multi-step expansion into the future; TD / Bellman form; weighted average of A / normalized weights, lambda-return<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;(1-lambda) * geometric series * discount series, infinite; lambda in 0-1 determines the main range of steps<br>
&emsp;&emsp;TRPO and clipping to 0.8-1.2, adaptive GAE length

&emsp;&emsp;Group Relative Policy Optimization (**GRPO**)<br>
&emsp;&emsp;No value network; multiple answers sampled for one question / rewards N(0,1) used as the relative advantage b; the objective function averaged over the multiple answers<br>
&emsp;&emsp;Better suited to a final answer or a single step

**DPO** training the reward model is the policy / in practice no reward model is needed<br>
&emsp;&emsp;Solving the optimization objective gives the form of the optimal policy; the reward function takes the form = policy ratio + Z; in the contrastive learning objective Z cancels out

Token-level loss: 1 / total length; sequence-level loss: 1 / sequence length, averaged over the sum of multiple groups

Fine-tuning / continued pre-training: pad sentences on the right; RL: pad on the left

**Deepseek R1**

R1-Zero: no fine-tuning / pure RL on v3-base<br>
&emsp;&emsp;GRPO: temperature-based random sampling to generate multiple answers<br>
&emsp;&emsp;With ground truth there is no need to train a reward model; the reward is the correctness of the result<br>
&emsp;&emsp;&emsp;&emsp;Format reward: the reasoning process is required to be placed between <think> tags, but the reasoning sentences themselves are not trained with a reward, preventing reward hacking where the reasoning scores very high but the result is wrong<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;A simple prompt template: system prompt / user prompt / assistant think-answer; generalization<br>
&emsp;&emsp;Self-evolution: the average thinking time gradually increases with training; the aha moment<br>
&emsp;&emsp;Readability and mixed languages

R1<br>
&emsp;&emsp;4 stages of training<br>
&emsp;&emsp;Cold-start data<br>
&emsp;&emsp;&emsp;&emsp;A few thousand samples / unlike SFT with its hundreds of thousands<br>
&emsp;&emsp;&emsp;&emsp;Long-CoT data generated by prompting R1-Zero with few-shot examples; based on the model's own distribution; the format of the reasoning process and the final answer<br>
&emsp;&emsp;Cold-start SFT on v3-base<br>
&emsp;&emsp;Reasoning RL: math and coding<br>
&emsp;&emsp;SFT<br>
&emsp;&emsp;&emsp;&emsp;Reasoning data: 600k samples; multiple answers sampled from the previous-stage model and filtered; v3 reviews the non-rule-based part<br>
&emsp;&emsp;&emsp;&emsp;Non-reasoning data: reused from v3, 200k samples<br>
&emsp;&emsp;&emsp;&emsp;Trained for 2 epochs<br>
&emsp;&emsp;RL: reasoning and non-reasoning<br>
&emsp;&emsp;&emsp;&emsp;Reasoning data / rule-based rewards<br>
&emsp;&emsp;&emsp;&emsp;General data / reward model; multiple answers ranked by preference; contrastive learning on each pair<br>
Benchmarks: MMLU, SimpleQA, SWE-bench Verified, LiveCodeBench, AIME 2024, etc.<br>
Dataset for distilling a large model to train a small model<br>

**Implementation of RL**

ijk

## Training & Inference Optimization

### Data, device, algorithm

Input and output length, number of inputs, throughput / data movement / weights and data<br>
GPU memory, transfer between memory and GPU cores / data movement / bandwidth, compute<br>
&emsp;&emsp;memory and compute caculations in section SFT and Pre-training<br>
Optimization (below)

**Quantization**

Numeric formats<br>
1 sign bit; 8 bits for the integer exponent with a bias offset; 23 bits for the binary fraction / decimal in [1, 2)<br>
Half-precision float 5/10; BF16 8/7<br>
INT8, INT4

&emsp;&emsp;INT8 = FP32 original value / scale factor s + zero point z<br>
&emsp;&emsp;Symmetric quantization: [-127, 127], z = 0<br>
&emsp;&emsp;Asymmetric quantization: [-128, 127]<br>
&emsp;&emsp;NF4 quantization / [0, 15] maps to [-1, 1], which maps to the normalized original value<br>
&emsp;&emsp;Number of data points * 4 + the scale factor stored as a 32-bit float

BF16: the default standard for mixed-precision training and saving of deep-learning parameters<br>
BF16 for full fine-tuning and inference; BF16 LoRA; NF4 QLoRA; Q4 has good compatibility / NF4 for inference

Quantization, low bit and integer arithmetic are fast<br>
&emsp;&emsp;Round to the nearest integer; when out of range, either clip or promote to a higher-precision bit width; the scale unit also has to take part in the arithmetic; nonlinear functions must be handled / lookup table, or linear approximation, or leave them unquantized<br>
&emsp;&emsp;&emsp;&emsp;The precision loss from dequantization does not affect the result much<br>
When inference<br>
Static quantization of the weights, dequantized at compute time<br>
Quantized computation over both weights and activations<br>
&emsp;&emsp;Dynamic quantization of activation inputs and outputs; static quantization of activations / computed in advance / faster inference<br>
The embedding and the task output module are not quantized; they share the vocabulary weights

Quantization formats<br>
safetensors by default<br>
bitsandbytes<br>
Apart from the outliers, the rounding during dequantization shows no visible change or difference<br>
Find the outliers; for int8, the outliers are not quantized and stay in fp16<br>
&emsp;&emsp;NF4 / an improvement over int4 / QLoRA<br>
&emsp;&emsp;Quantile quantization with a lookup table / matched against a Gaussian distribution / [-1, 1] mapped to [0, 15]<br>
&emsp;&emsp;Per group of seq; the fp32 scale factor is stored on disk or in memory; second-level quantization over the scale factors of n groups<br>
Inference only<br>
&emsp;&emsp;GPTQ: static quantization; block-wise, group-wise quantization / reduces the occurrence of outliers<br>
&emsp;&emsp;AWQ: GPTQ with per-layer quantization<br>
&emsp;&emsp;llama.cpp: GGUF, a single-file format, not the general ones above; ollama supports GGUF only

**Distributed parallel** GPUs training and inference

Data parallel<br>
&emsp;&emsp;DP: the data is split across GPUs, the model is replicated, and the gradients are summed and broadcast back to every replica<br>
&emsp;&emsp;&emsp;&emsp;The primary GPU 0 sends and receives (n-1) units of data each way; each non-primary GPU handles 1 unit<br>
&emsp;&emsp;Distributed DDP: because of GPU waiting and communication, the gradient results are partitioned across the number of GPUs — ring-allreduce<br>
&emsp;&emsp;&emsp;&emsp;Compared with DP's parameter-server communication: bandwidth, ring topology, load<br>
&emsp;&emsp;&emsp;&emsp;Communication time / volume is independent of the number of GPUs; in each stage every card does n-1 steps on 1/n of the data, in parallel<br>
&emsp;&emsp;Gradient bucketing: gradients are computed and propagated layer by layer during backpropagation, instead of waiting for all of them to finish<br>
&emsp;&emsp;DeepSpeed ZeRO (zero redundancy optimizer)

Forward and backward passes are computed independently on each card; the gradient results are communicated<br>
&emsp;&emsp;Communication without memory partitioning; allreduce sums and replicates<br>
&emsp;&emsp;&emsp;&emsp;One implementation is ring-allreduce, split into the reduce-scatter and allgather stages - partitioned

Mixed-precision GPU memory<br>
Model weights 32 - replica 16 - forward activations (temporary) - loss - backward gradients 16 (temporary) - gradients 32 (temporary) - optimizer (optimizer states 32, model weights 32, updated)

ZeRO-1: the optimizer is averaged; DDP holds n full copies of the weights / n-1 copies are redundant; the parameters are divided evenly by n, each part placed in its corresponding partition, all gather<br>
ZeRO-2: on top of 1, for the gradients the allgather stage of ring-allreduce is removed<br>
ZeRO-3: on top of 2, model partitioning with communication under data parallel / not model parallel<br>
&emsp;&emsp;data parallel with model parallel / but the complete model can still be held / partitioned, with the complete model present temporarily / communication happens as the computation proceeds<br>
From 1 to 3: memory savings, communication overhead, and implementation complexity all increase

Model / parameter-weight parallel: each card holds only a part / the parameter weights are not communicated<br>
&emsp;&emsp;Tensor parallel (TP): intra-layer parallel / a single card cannot fit one layer, so the weight matrix is split<br>
&emsp;&emsp;&emsp;&emsp;Linear algebra composed of rows / columns: row parallel / column parallel, with the results aggregated<br>
&emsp;&emsp;Pipeline parallel (PP): layers and batches in series / in time order / output values passed along; the non-triangular region achieves a parallel effect<br>
<br>
3D parallel: data, tensor, pipeline<br>
<br>
Sequence parallel: the sequences communicate with each other during attention computation<br>
<br>
MoE expert parallel (EP): the different experts are independent / naturally parallel across cards, all-to-all communication<br>

**FlashAttention**

IO acceleration / operates in the fast SRAM, reducing HBM reads and writes and lowering the pressure on memory capacity and bandwidth<br>
Memory (DRAM) / GPU memory (GDDR) - inside the GPU package - GPU memory (HBM) - inside the GPU core die - cache (SRAM) - compute cores CUDA and Tensor cores<br>
&emsp;&emsp;SRAM, HBM, DRAM: communication cost decreases and capacity increases along this order<br>
&emsp;&emsp;Memory or DRAM, GPU memory is essentially DRAM, while the CPU's sits outside the package<br>
&emsp;&emsp;&emsp;&emsp;RTX GDDR, H100 / A100 HBM<br>
&emsp;&emsp;Cache means SRAM; both CPUs and GPUs have it; CPU: L1 / L2 / L3; GPU: L0 / L1, shared memory, L2<br>
&emsp;&emsp;&emsp;&emsp;CPU: large cache, few registers; GPU: small cache, a huge number of registers<br>
&emsp;&emsp;RAM: SRAM, DRAM<br>
&emsp;&emsp;Flash memory / flash / SSD

Attention operates on small sequences one at a time, transferred within SRAM<br>
&emsp;&emsp;But softmax needs the complete sequence<br>
&emsp;&emsp;Safe softmax: exp(x - max)<br>
&emsp;&emsp;In autoregressive order, maintain lists of x and max

Bandwidth: the data-movement speed, the number of tokens transferred per unit time<br>
&emsp;&emsp;With a lot of data you have to wait for the transfer<br>
During generation the computation for a single token is tiny; all the data is moved from video memory to the compute units, and that time is far longer than the compute time<br>
&emsp;&emsp;Prefill computation: the time to the first token, TTFT / time to first token<br>
&emsp;&emsp;Generation, decode: the output TPOT / time per output token, or the interval ITL / inter-token latency<br>

PyTorch SDPA (scaled dot-product attention) automatically selects the suitable computation<br>
&emsp;&emsp;flash-attn: conditional and requires installation; memory-efficient: block-wise computation, requires installing xformers; math: built in<br>
The transformers library requires it to be specified at model loading time: either flash-attn (requires installation) or PyTorch SDPA<br>
DeepSpeed does not implement it itself; a parameter has to be set as a flag, and other libraries are still needed to enable it<br>
accelerate: no direct relation<br>
vLLM: available by default, no need to enable it, no need to install flash-attn

**PagedAttention** vLLM

GPU memory problems<br>
&emsp;&emsp;Memory is allocated according to the maximum length, but the actually generated length is far shorter<br>
&emsp;&emsp;The allocated memory is not used yet, and gets used gradually<br>
&emsp;&emsp;Memory is allocated contiguously; fragments are left idle; not enough left to allocate

The virtual-memory paging mechanism of the operating system: the KV cache is split into fixed-size blocks of 16 tokens, and a page table (block table) manages where the KV tokens are stored, making the layout non-contiguous and eliminating memory fragmentation<br>
Shared KV cache / blocks: scenarios where the beginnings of the sequences are the same

Inference frameworks such as vLLM: quantization, PagedAttention, distributed multi-GPU parallelism, FlashAttention, concurrency

**Summary**<br>
Quantization, KV cache PagedAttention, distributed parallel GPUs, FlashAttention<br>
Concurrency optimization<br>
&emsp;&emsp;Continuous batching: as soon as a request finishes it is removed and a new request is added<br>
&emsp;&emsp;Prefill / decode separation<br>
Same prefill cache<br>
Context and memory management<br>
Streaming output

**Implementation of distributed trainning accelerate deepspeed**

Using deepspeed with accelerate requires installing deepspeed first, and deepspeed requires CUDA toolkit<br>
<br>
accelerate config or yaml, huggingface documentation and github examples<br>
&emsp;&emsp;deepspeed config and json<br>
torch.utils.data's TensorDataset inherits Dataset, packaging in-memory tensor data into tuples<br>
model, dataloader, optimizer = accelerator.prepare(model, dataloader, optimizer)<br>
...change loss.backward to accelerator.backward(loss)<br>
<br>
Launch from command line accelerate launch --config_file ./xxx.yaml train.py<br>
Display GPU memory with nvidia-smi<br>
<br>
@dataclass defines a class with built-in initialization, etc.<br>
dataclasses library .field() some special requirements for fields<br>
parser = HfArgumentParser(dataclassarg, TrainingArguments) hugging face command line argument parser not instantiated<br>
args, training_args = parser.parse_args_into_dataclasses() read command line arguments and instantiate<br>
&emsp;&emsp;parser can .add_argument, but not in dataclass, the latter is recommended<br>
Command line instructions and parameters .sh shell file<br>
&emsp;&emsp;Linux terminal autodl chmod for execution permission and nohup for background running, Windows does not have this

**Implementation of vLLM**

ijk

### API

### Function call

### Prompt

**Prompt engineering**

System prompt, user prompt: clear and specific requirements for the model<br>
In-context learning (ICL) / diversity of the pre-training data and of the model parameters; few-shot learning; zero-shot learning<br>
Chain of thought (CoT): the samples show thinking and reasoning; add a line saying think step by step

CoT inspires the model rather than few-shot right/wrong examples; task planning<br>
Search<br>
&emsp;&emsp;Tree of thoughts (ToT): decompose into one step of thinking, candidate generation, evaluation by the large model / testing / voting on candidates, search algorithm<br>
&emsp;&emsp;Breadth-first search (BFS), depth-first search (DFS)<br>
&emsp;&emsp;Self-consistency: multi-path generation / top-k, top-p and temperature; vote for the one that appears most frequently

## Applications

### Multimodal

Multimodal perception / input; a video is composed of image frames<br>
&emsp;&emsp;Converting each modality into a text description / a dedicated model; information is lost<br>
&emsp;&emsp;Native multimodality: an encoder for each modality + modality fusion + a large-model decoder; heavy computation, and modality fusion is difficult<br>
Multimodal action<br>
&emsp;&emsp;Screen and UI operation / screenshots as input<br>
&emsp;&emsp;Text, external calls or other models, diffusion-based generation<br>
Modality fusion -> flattening -> fully connected layer

**ViT**<br>
The pixels of a small patch are treated as a token; projection aligns them to dim; positional encoding / learnable<br>
Add a class token / finally take the dim of the first one; global average pooling (GAP) / seq_len to 1<br>
ViT positional encoding (1, N+1, D); N+1 because there is one extra cls token carrying the class features

**CLIP**

Data: 400 million text-image pairs<br>
Contrastive learning on the similarity / difference between the text and image vectors<br>
&emsp;&emsp;The vector output by the text encoder / BERT, and the vector output by the image encoder / ViT<br>
&emsp;&emsp;The similarity matrix of n texts and n images is n*n; the target matrix has 1 on the diagonal and 0 elsewhere<br>
&emsp;&emsp;&emsp;&emsp;Cross-entropy against the target, bidirectional: text to image, and image to text<br>
Multiple images in one batch

At inference, a natural-language prompt can improve and influence the image matching, no longer fixed labels<br>
Using CLIP: fine-tune by adding a new linear layer for the classification output, or fine-tune the whole model in full

**LLaVA**

Image-caption text data augmentation: based on the image and the caption, an existing large model raises questions and answers them in detail<br>
&emsp;&emsp;The system prompt in slides<br>
&emsp;&emsp;The types of augmented, generated QA: conversation 58k, detailed description 23k, complex reasoning 77k

Model structure: image encoder - a learnable projection for alignment / a linear layer is the simplest, text input, large model<br>
Two stages of training<br>
&emsp;&emsp;Freeze the image encoder (CLIP's ViT) and the large model except for the final output; pre-train the projection layer<br>
&emsp;&emsp;&emsp;&emsp;Training data: CC3M, image-text pairs<br>
&emsp;&emsp;&emsp;&emsp;Loss on the text input and output<br>
&emsp;&emsp;&emsp;&emsp;There is an <image> placeholder, which is replaced by the feature sequence when the model processes it<br>
&emsp;&emsp;Freeze the image encoder (CLIP's ViT); instruction-tune the projection layer and the large model<br>
&emsp;&emsp;&emsp;&emsp;QA data augmented and generated by GPT-4: images, multi-turn QA<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;The first turn takes the image and the question, or the question and the image — the order does not matter<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Later turns take the question only<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;The conversation history is concatenated with <stop> as the training input<br>
&emsp;&emsp;&emsp;&emsp;ScienceQA dataset: single-turn QA with the reasoning process + the result

LLaVA-1.5: more datasets added / yes-no image questions; besides the cropped image, the encoder also takes the original whole image

**Multimodel data**

Contrastive learning: the text / vision encoder, BERT / ViT trained from scratch, image-text pairs, hundreds of millions to billions of samples<br>
Pre-training: cross-modal alignment, the projection layer, image-text pairs, millions to tens of millions of samples<br>
SFT: language ability, QA ability, the projection layer + LLM, image - multi-turn QA<br>
&emsp;&emsp;Full fine-tuning: thousands to hundreds of thousands of samples<br>
&emsp;&emsp;LoRA for users: thousands of samples<br>
For a specific category or domain, the number of training samples drops accordingly

**Implementation of LLaVA**

Define the train function within one epoch<br>
Define the learning rate cosine annealing, progressing with the timestep, a function of the timestep<br>
batch, image_nums, 3, len, wide: the image processor tensor

Define the function for model loading<br>
Freeze the model parameters: .requires_grad=False

Model<br>
clip.vision_model<br>
The features replace the <image> placeholder

Pre-training for alignment: epochs=1, batch=256-512<br>
SFT loads the pre-trained model

llava.model

### RAG

### Agent

### Engineering and project

**Code notice**

pass: an empty placeholder<br>
yield: produces a value and continues / differs from return<br>
while: stop condition: iteration condition<br>
break exits the loop; continue ends the current iteration and moves on<br>
Backslash \: line continuation; escape characters<br>
<br>
max / argmax: the size of the specified dimension is reduced to 1; whether to keep that dimension, default keepdim=False<br>
<br>
\*a / in a function definition: extra positional arguments / packed into a tuple; at a function call: a list or tuple unpacked into positional arguments / passed in order<br>
\*\*a / in a function definition: extra keyword arguments / packed into a dict; at a function call: a dict unpacked into keyword arguments / matched by argument name<br>
<br>
A yaml file loaded into a dict; json<br>
<br>
a[] tensor indexing, dict indexing, list indexing, tuple indexing / a tuple cannot be modified<br>
The set type set(); zip(,,) over different iterables forms an iterable of tuples; enumerate(x) returns an iterable of tuples of the index and the element of x<br>
[,] lists, arrays

Tensor dimension operations<br>
<br>
A tensor with requires_grad=True: gradient storage and whether it is differentiable<br>
nn parameters are enabled by default<br>
Calling .backward() multiple times accumulates the gradients<br>
<br>
Function decorator @; temporary resource management with f():; gradients: torch.inference_mode (stricter) / torch.no_grad<br>
Separate: model.eval() / e.g. turning off dropout<br>
<br>
A forward pass through the network stores the history and the outputs<br>
<br>
Hyperparameters formatted with argparse<br>
def main(args)<br>
main(opt)

@staticmethod<br>
The function is not bound to the class or the instance, cannot be inherited, and cannot access the variables in the class via self / it can access them explicitly; both the class and the instance can call the function via self<br>
<br>
A function defined inside a function can use the variables of the outer function<br>
<br>
super()._ init _: when the inherited class takes arguments, they have to be passed in, or the remaining arguments are passed as a dict<br>
Keep the parameters of initialization and instantiation separate from the parameters of the class's other functions<br>
<br>
Variables defined in a Python class must have a value, or be initialized with a value in _ init _, or receive a value at instantiation<br>
dataclasses and pydantic allow a variable to have only a type annotation without a value

os.environ['key']='' reads environment variables; the api key differs across platforms; different virtual environments do not interfere<br>
&emsp;&emsp;Whether the variables are public when the project is open-sourced

epoch: the number of times the dataset is trained on<br>
Memory usage is shown by nvidia-smi<br>
<br>
enumerate(dataloader) adds its own id / timestep<br>
&emsp;&emsp;for step, batch in<br>
&emsp;&emsp;x, y = batch<br>
for pg in optimizer.param_groups: the optimizer's parameter groups, used to update the learning rate<br>
&emsp;&emsp;pg['lr'] = lr<br>
<br>
with ctx, the short form of with ...:, mixed precision 16, fp16 / bf16 chosen automatically according to the device or specified manually<br>
&emsp;&emsp;Forward pass, loss<br>
Backward pass, gradients; the loss has to be scaled up to keep the gradients from falling below the 16-bit lower limit<br>
Update after the accumulated number of steps<br>
The Trainer in the transformers library specifies mixed precision










