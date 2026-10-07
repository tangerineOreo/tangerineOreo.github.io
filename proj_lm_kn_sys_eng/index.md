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

transformers<br>
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

utils.data.dataset: an abstract base class, _ getitem _ has no fixed format<br>
&emsp;&emsp;A processor can be passed in<br>
DataLoader has no requirements on the dataset format, and processes on the fly: DataLoader(dataset, ..., collate_fn / collator)<br>
&emsp;&emsp;The default / without a collate_fn, only the supported formats get stacked from samples into tensors<br>
datasets.Dataset has more features<br>
&emsp;&emsp;After .map the data can be saved<br>
&emsp;&emsp;When used with the default DataLoader, mind the format: dicts and tensors<br>
Trainer is equivalent to a DataLoader in a loop; internally it calls the DataLoader / through train_dataset and the various data_collators, etc.<br>
<br>
PreTrainedModel<br>
&emsp;&emsp;Inherits from nn.Module, supports DataLoader<br>
&emsp;&emsp;Supports Trainer; returns more than nn.Module / has loss, etc.; has a generate function<br>
&emsp;&emsp;Rich supported features and parameters; ecosystem support / across the various frameworks and libraries<br>
&emsp;&emsp;Bound to a config; loaded as MyCustomModel together with MyCustomConfig; AutoModel & AutoModelFor... and AutoConfig load it correspondingly after registration

AutoModel choice / pre-trained model / no task output / base / BertModel<br>
AutoModelFor / pre-trained model / has a task output module but no input module<br>
&emsp;&emsp;CausalLM decoder, Seq2SeqLM encoder-decoder, others<br>
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
tokenizer_template(message, add_generation_prompt: needed for inference / for training the answer is already there and gets concatenated<br>
&emsp;&emsp;tools<br>
&emsp;&emsp;tokenizer=true, pt, return_dict gives the attention_mask<br>
&emsp;&emsp;Or apply the template first and then pass it to the tokenizer, which also has other parameters<br>
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

Verifiable rewards<br>
&emsp;&emsp;Math, code passing the tests, formatted output<br>
Open-source reward models; using a large model as the reward model<br>
Preferences scored by a large model, preferences scored by machine learning or by rules, mixed with human scoring, to train the reward model<br>
<br>
Training complexity is several times that of SFT: PPO 3-5x or more, DPO / GRPO 1.5-3x; the reward model, the AC (actor-critic) model<br>
Preference-pair data, data volume 0.1-0.3 of SFT<br>
<br>
KL(p,q) relative entropy: summing p * (log p - log q) = the cross entropy of p and q - the entropy of p<br>
&emsp;&emsp;E_p[log p - log q]<br>
log_softmax(p) - log_softmax(q)<br>
Jensen-Shannon<br>
<br>
The policy gradient is the expectation over the whole trajectory, or over state-action pairs<br>
PPO and GRPO are over timesteps or sequence steps<br>
Essentially both are over the trajectory

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

tokenizer.pad_to_left<br>
<br>
AutoModelForCausalLM, AutoTokenizer<br>
message<br>
tokenizer_template(message<br>
model.to(device), ids_input.to(device)<br>
model.generate(ids_input)<br>
tokenizer.batch_decode(ids_output)<br>
<br>
datadict = datasets.load_dataset<br>
Process the data<br>
<br>
wandb.ai/site<br>
import wandb<br>
wandb.login(key='')<br>
wandb.init(project='')

GRPO: number of samples 8, a high sampling temperature coefficient<br>
The format of the final result has to be consistent: 1/2 vs. 0.5<br>
<br>
system prompt<br>
def the rule-based reward function<br>
<br>
GRPOConfig(): inference sampling and training, vllm parameters, the RL learning rate 5e-6 is relatively small, 1 epoch<br>
GRPOTrainer() with reward_funcs

PPO dataset: 2 sft, 4 reward-model, 4 rlhf<br>
Freeze parameters, learning rate, LoRA<br>
bf16 mixed precision, accelerate.Accelerator<br>
<br>
reward model: the large model / ref model with a linear layer added at the end to output a score<br>
Can be trained with trl<br>
<br>
value model: the large model / ref model with a linear layer added at the end to output a score<br>
There are other models for this now as well<br>
Freeze parameters, learning rate, LoRA<br>
Train two models: the policy and the value model

**verl** suitable for larger-scale scenarios, distributed, each model declared independently<br>
&emsp;&emsp;ray, vllm<br>
grpo_config.yaml<br>
&emsp;&emsp;The algorithm, data-related settings and the reward function, actor, inference, trainer training parameters<br>
parquet data format<br>
Reward function definition; trl / verl do not train a reward model

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
MoE expert parallel (EP): the different experts are independent / naturally parallel across cards, all-to-all communication

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

Inference frameworks such as vLLM support multiple GPUs and multiple models<br>
&emsp;&emsp;Multiple GPUs with multiple instances and ports; a single GPU with multiple instances and ports<br>
&emsp;&emsp;vllm-router: one port, multiple models<br>
&emsp;&emsp;The same base model + multiple LoRAs<br>
&emsp;&emsp;Offline inference / without enabling HTTP: model_a/b = vllm.LLM(model='') to load the model<br>
<br>
Multi-threading is limited by Python's GIL; multi-processing blows up the memory; single-threaded asynchronous scheduling<br>
Waiting for a result: async / sync; multiple at the same time: concurrency, not parallelism / multi-process parallelism<br>
When a synchronous function has to wait for the result: asyncio.to_thread<br>
<br>
gradient_checkpointing: trading time for space; during training, activations are computed on the fly and only partially saved

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

Install the CUDA toolkit with conda: when writing CUDA C++, or when installing flash-attention / xformers / deepspeed and other low-level packages that need to be compiled<br>
&emsp;&emsp;The PyTorch installation includes cuDNN and the CUDA runtime, but not the compilation part

pip install vllm<br>
&emsp;&emsp;vLLM is asynchronous by nature; internally it uses asyncio + FastAPI and the OpenAI API<br>
&emsp;&emsp;The API is the HTTP-layer protocol / HTTP has no sync-vs-async distinction, it is irrelevant<br>
&emsp;&emsp;Sync vs. async in OpenAI refers to the client<br>
Running the local model with vLLM<br>
&emsp;&emsp;On the server terminal: vllm serve \ --model path \ --served-model-name \ --port \ — the backslash continues the line<br>
&emsp;&emsp;&emsp;&emsp;vllm serve = python -m vllm.entrypoints.openai.api_server<br>
&emsp;&emsp;On the client, in an IDE file: client = openai(...); the url supports localhost v1, and api_key can be filled with any non-empty string<br>
&emsp;&emsp;&emsp;&emsp;.create(model model-name, message, ...)<br>
&emsp;&emsp;Or curl from a non-server machine, with OpenAI parameters<br>
Or vllm.LLM and SamplingParams to load the model path and control the generation<br>
&emsp;&emsp;When vLLM is not needed, other libraries can also load and use the model

curl url/models to view the list of models<br>
&emsp;&emsp;Deploying multiple models requires Docker packaging

Outside the server<br>
AutoDL instance - custom service; ssh on Windows, Linux / Mac, with a password; on the command line, local port 8000 maps to the server's localhost:xxx<br>
&emsp;&emsp;curl url/models<br>
An enterprise-verified domain name; users use the url directly

**Concurrency**

The number of requests being executed at the same moment<br>
&emsp;&emsp;Average requests per second * the average response time per request<br>
Layers<br>
&emsp;&emsp;The API gateway: the traffic entry point<br>
&emsp;&emsp;Load balancing: distributing the traffic to thousands of GPU nodes / each node has multiple GPUs<br>
&emsp;&emsp;The inference layer: quantization / FlashAttention / KV cache with PagedAttention / continuous batching / prefill-decode separation<br>
Concurrency estimation and stress testing<br>
The vLLM api and concurrency parameter settings<br>
Adding a layer of nginx in front / concurrency limiting and messaging / encryption / distribution<br>
<br>
Stress testing<br>
TTFT, TPOT, throughput, concurrency, completion rate / number completed over the total number of requests<br>
Input patterns<br>
&emsp;&emsp;A fixed number of requests / like continuous batching, always keeping a fixed number of requests in flight; gradual ramp-up of the load<br>
&emsp;&emsp;A fixed rate / e.g. 100 requests per second<br>
&emsp;&emsp;Combinations of input and output lengths<br>
Tools<br>
&emsp;&emsp;vLLM: random / ShareGPT conversation records / sonnet long text / BurstGPT traffic spikes, and other datasets<br>
&emsp;&emsp;evalscope perf: random / the openqa dataset<br>
&emsp;&emsp;python locust<br>
<br>
Stress testing a function<br>
&emsp;&emsp;from locust import HttpUser, User, task, between<br>
&emsp;&emsp;class MyFunctionUser(User):<br>
&emsp;&emsp;wait_time = between(1, 3)<br>
&emsp;&emsp;@task<br>
&emsp;&emsp;def call_my_function(self):<br>
&emsp;&emsp;Test with fixed inputs<br>
VLLM stress-tests the OpenAI-compatible api over http; it cannot stress-test a function<br>
Stress-testing a model and stress-testing a function differ a lot in timing: retrieval, the agent flow, tool-calling time, the number of model calls<br>
<br>
Related factors<br>
&emsp;&emsp;Inference: prefill and decode / data, devices, algorithms<br>
&emsp;&emsp;Memory size, data movement / bandwidth, compute<br>
&emsp;&emsp;Data volume: input and output length, number of inputs, throughput / data movement / weights and data<br>
&emsp;&emsp;Algorithm optimization: quantization / FlashAttention / KV cache with PagedAttention / continuous batching / prefill-decode separation / caching for identical prefixes<br>
&emsp;&emsp;Streaming output<br>
&emsp;&emsp;RAG retrieval, the agent flow<br>
<br>
Caculation: number of requests and time<br>
&emsp;&emsp;Requirements: 500 * average online ratio 0.2-0.4; average daily requests 5/24; peak hourly requests 6/3600s; response time 5 or 10 s<br>
&emsp;&emsp;Memory: kv cache = official / vllm 0.9 * RTX 4090 - model weights, quantized, and the shared vocabulary<br>
&emsp;&emsp;Max concurrency = number of kv tokens / average tokens per request; number of kv tokens = kv cache / memory per token<br>
&emsp;&emsp;1 token * 2 (kv) * 36 layers * 1024 total dim * 2 Bytes = 144KB<br>
&emsp;&emsp;Context 1024-8192 / 32k tokens; concurrency 120-15 / 4<br>
&emsp;&emsp;Stress test: concurrency 60, TTFT = 5 s

### API

client = openai() with api_key and base_url<br>
output = client.chat.completions.create<br>
&emsp;&emsp;The message style: role, content<br>
&emsp;&emsp;output.: several answer candidates, choices[0].message.content<br>
&emsp;&emsp;Sampling generation: max number of tokens / temperature / top-p / top-k / repetition penalty

Streaming output: stream=true, for chunk, for event in stream

Structured output<br>
&emsp;&emsp;response_format with a json schema<br>
&emsp;&emsp;&emsp;&emsp;Written by hand, or fm.model_json_schema()<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;fm is a class based on pydantic.BaseModel: annotated types, pydantic.Field descriptions<br>
&emsp;&emsp;Alibaba Cloud Model Studio (Bailian) output format settings; writing the format in the prompt

Some models, e.g. reasoning models, do not support function call, json output, generation control

The chat completions protocol<br>
&emsp;&emsp;message<br>
&emsp;&emsp;tool is custom, in schema format<br>
&emsp;&emsp;Executed and added manually<br>
&emsp;&emsp;Generally compatible across models<br>
The responses protocol<br>
&emsp;&emsp;input, instruction<br>
&emsp;&emsp;The output is items, including message and the intermediate multi-step process: planning, executing, results, reflecting, looping<br>
&emsp;&emsp;tool: custom + built-in + MCP server; in schema format or by built-in name; built-in web search / file search / code interpreter / computer use<br>
&emsp;&emsp;Completes the process automatically<br>
&emsp;&emsp;Some model vendors are not compatible; vLLM and others are not compatible<br>
The tool schema differs between the two protocols

### Function call

tools: function call<br>
Whether a function needs to be called: no / text output; yes / call the function<br>
The tools list, the tool dict<br>
&emsp;&emsp;type function; name; description: a detailed description; parameters / type / properties / required<br>
Manually add the conversation history to the message<br>
&emsp;&emsp;role assistant with message.tool_calls as a list of dicts; parse and extract the function name and arguments; the execution result; role tool with the result format; role assistant with message.content as the answer; content and tool_calls are mutually exclusive

&emsp;&emsp;transformers library loads model<br>
&emsp;&emsp;input_text = tokenizer.apply_chat_template(message, tools=tools,<br>
&emsp;&emsp;model generation format comes from the system prompt<br>
&emsp;&emsp;tokenizer.decode<br>
&emsp;&emsp;Parse whether it is a tool call or a text output, and extract the function name and the arguments

&emsp;&emsp;It is recommended to use the API, or the API service of an inference framework<br>
function and the arguments / equivalent to rewriting the input, or producing an output

### Prompt

System prompt, user prompt: clear and specific requirements for the model<br>
In-context learning (ICL) / diversity of the pre-training data and of the model parameters; few-shot learning; zero-shot learning<br>
Chain of thought (CoT): the samples show thinking and reasoning; add a line saying think step by step

CoT inspires the model rather than few-shot right/wrong examples; task planning<br>
Search<br>
&emsp;&emsp;Tree of thoughts (ToT): decompose into one step of thinking, candidate generation, evaluation by the large model / testing / voting on candidates, search algorithm<br>
&emsp;&emsp;Breadth-first search (BFS), depth-first search (DFS)<br>
&emsp;&emsp;Self-consistency: multi-path generation / top-k, top-p and temperature; vote for the one that appears most frequently

Prompt engineering framework<br>
Components: instruction, context / background, output format<br>
Enhancement techniques: few-shot, CoT, self-consistency

Writing: symbols, tags, the all-purpose-yet-not-all-purpose template .md<br>
&emsp;&emsp;Task instruction<br>
&emsp;&emsp;Background context<br>
&emsp;&emsp;The input data to be processed<br>
&emsp;&emsp;Requirements<br>
&emsp;&emsp;Output format and structure template<br>
&emsp;&emsp;Examples<br>
&emsp;&emsp;The reasoning process

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

Up-to-date information, professional document libraries to reinforce professional standards, enterprise local data<br>
Requires customization, and has to rely on the model's own answering ability<br>
Reduce hallucination

Document parsing - chunking - vectorization / indexing / storing into the database<br>
query - embedding - retrieval matching - retrieval results<br>
The retrieval results serve as the context for generating the answer

Without building a RAG: code, personal knowledge and experience, etc. — lightweight<br>
Retrieval speed: graph RAG < tree RAG < RAG<br>
100 papers, 3000 vectors — a very small amount of data; below 100k vectors the performance difference is negligible

Chunking effectiveness<br>
&emsp;&emsp;Affects semantic completeness<br>
&emsp;&emsp;The smaller the chunk, the more precise the similarity<br>
Rule-based chunking: fixed length, headings and paragraphs, complete sentences with overlap<br>
Semantic chunking<br>
&emsp;&emsp;Compare the similarity of two chunks; merge them if the semantics are similar<br>
&emsp;&emsp;Add or remove a sentence and compare the similarity, to tell whether it is a semantic boundary<br>
&emsp;&emsp;Structured information as triples of subject-predicate-object, forming a knowledge graph<br>
Rule-based chunking is generally used; semantic chunking takes a long time

The role of metadata: fast filtering by time, url, keywords<br>
Vector databases: nearest-neighbor clustering, quickly locating a similar region, which makes retrieval easier<br>
Normalized dot product for high-concurrency RAG scenarios; Euclidean distance similarity for non-text scenarios<br>
Indexes<br>
&emsp;&emsp;HNSW: hierarchical navigable small world<br>
&emsp;&emsp;IVF: clustering in high-dimensional space, k-means clusters<br>
&emsp;&emsp;PQ

**LlamaIndex**

Components<br>
&emsp;&emsp;Different data sources: pdf / txt / documents, SQL databases, HTML web pages, audio and video, json<br>
&emsp;&emsp;Data connectors / converting to text: SimpleDirectoryReader, DatabaseReader, WebPageReader<br>
&emsp;&emsp;Document processing: splitting, embedding, metadata / tagging with some keywords<br>
&emsp;&emsp;Adding indexes: VectorStoreIndex, ListIndex, SummaryIndex, KeywordTableIndex, TreeIndex, KnowledgeGraphIndex<br>
&emsp;&emsp;Storage: redis, chroma, etc.<br>
&emsp;&emsp;The query engine and the large model

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader<br>
reader = SimpleDirectoryReader — document parsing<br>
&emsp;&emsp;file_extractor: the parser, optional<br>
documents = reader.load_data()<br>
for x in documents — a list<br>
<br>
LlamaHub is open source<br>
LlamaCloud: advanced parsing, api key, free quota + paid<br>
MinerU, Marker, DeepDoc based on RAGFlow<br>
<br>
from llama_index.readers.web import SimpleWebPageReader; the .readers.xxx packages need to be installed<br>
Or install LlamaHub; core.download_loader() downloads from LlamaHub<br>
parser = llama_cloud_services.LlamaParse — optionally parses into .md<br>
&emsp;&emsp;Used in file_extractor

Splitting with a splitter, creating node objects<br>
from llama_index.core.node_parser import SentenceSplitter<br>
splitter = SentenceSplitter<br>
nodes = splitter.get_nodes_from_documents(documents)<br>
&emsp;&emsp;nodes: json dicts with file attributes, metadata fields, etc.

from llama_index.vector_stores.faiss import FaissVectorStore<br>
faiss_index = faiss.IndexFlatL2(d) — Euclidean distance, L2 similarity computation<br>
vector_store = FaissVectorStore(faiss_index)

storage_context = StorageContext.from_defaults(vector_store)<br>
index = VectorStoreIndex(nodes, storage_context)<br>
&emsp;&emsp;retriever = index.as_retriever(...) or VectorIndexRetriever(index, ...)<br>
&emsp;&emsp;&emsp;&emsp;renodes = retriever.retrieve('')<br>
&emsp;&emsp;engine = RetrieverQueryEngine.from_args(retriever); there is no RetrieverChatEngine<br>
engine = index.as_query_engine() — internally it also calls as_retriever<br>
&emsp;&emsp;.as_chat_engine(ChatMemoryBuffer for multi-turn conversation, system prompt)<br>
response = engine.query / chat('')<br>
&emsp;&emsp;print(response)<br>
Streaming output: as_query/chat_engine(stream_response=True), or engine.stream_query / chat<br>
&emsp;&emsp;for chunk in response.response_gen: print(chunk)

**BGE Milvus**

LlamaIndex / LlamaCloud for parsing, LlamaIndex for chunking, BGE for vectorization, Milvus as the database<br>
&emsp;&emsp;BGE in LlamaIndex gives only dense vectors / BGE in pymilvus / essentially the official BGE<br>
&emsp;&emsp;bm25 with Milvus in LlamaIndex / more hybrid retrieval with pymilvus / the parameters are not compatible but the client can be shared<br>
Hybrid retrieval: term-frequency sparse; semantic sentence vectors with term importance, sparse; semantic sentence vectors, dense; semantic word vectors, Multi-vector<br>
<br>
Define a function to chunk the parsed markdown document — courseware<br>
&emsp;&emsp;Split paragraphs by line breaks, detect the start of a markdown table, detect level-1 / 2 / 3 / 4 headings<br>
&emsp;&emsp;chunks.append<br>
The chunk size in LlamaIndex is counted in characters, while the embedding model counts tokens: Chinese 1.5-2 characters/token, English 4 characters/token<br>
<br>
pip install pymilvus[model]<br>
milvus.io/docs/zh<br>
<br>
Download BAAI/bge-m3 locally with the huggingface_hub library or from the web page, or import the official package and let it be cached automatically on the system at runtime<br>
from milvus_model.hybrid import BGEM3EmbeddingFunction<br>
model = BGEM3EmbeddingFunction(path, device) loads the model<br>
chunks_embedding = model(chunks) gives sparse and dense vectors; chunks is the list of vectors

Connecting to Milvus, the format template for creating a collection<br>
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, Function / BM25, etc.<br>
&emsp;&emsp;MilvusClient(uri=path or the api url)<br>
&emsp;&emsp;&emsp;&emsp;'./milvus.db'<br>
&emsp;&emsp;fields = [FieldSchema(name='id or pk/text/vector/dense_vector/sparse_vector', data type, other arguments),]<br>
&emsp;&emsp;schema = CollectionSchema(fields=fields<br>
&emsp;&emsp;collection_name = ''<br>
&emsp;&emsp;co = Collection(name=collection_name, schema=schema)

Add the index before inserting the data, to save the cost of rebuilding it<br>
index_params = {'index_type':'', 'metric_type':''}<br>
&emsp;&emsp;index_type: generally HNSW; when index_type=AUTOINDEX it is chosen automatically; for sparse it is SPARSE_INVERTED_INDEX<br>
&emsp;&emsp;metric_type: the similarity computation — BM25 for term frequency, COSINE, L2, IP<br>
&emsp;&emsp;&emsp;&emsp;They are not the same; there are also different processing steps and structures, such as normalization<br>
&emsp;&emsp;&emsp;&emsp;For sparse: BGE-M3 uses IP, BM25 uses BM25; cosine is not supported and is meaningless, and L2 has the cost of computing over the dimensions with 0 values<br>
co.create_index(field_name='vector/dense_vector/sparse_vector', index_params)<br>
co.load() into memory; persistent storage is not json / list but binary files<br>
<br>
Adding the data in batches, several chunks per batch<br>
datachunks = [the field order in the template]<br>
co.insert(datachunks)

query_embedding = model(query)<br>
&emsp;&emsp;query_embedding['dense'][0] — the query vector in the list<br>
&emsp;&emsp;query_embedding['sparse'][0] — the query rows of the matrix<br>
Retrieving the text<br>
dense/sparse_results = collection.search( — in the new version, client.search<br>
&emsp;&emsp;data=query_embeddings['dense'],<br>
&emsp;&emsp;anns_field='dense_vector',<br>
&emsp;&emsp;param={'metric_type':''},<br>
&emsp;&emsp;limit=10 — top-k,<br>
&emsp;&emsp;output_fields=['text',...] — the field names to be returned<br>
)<br>
dense/sparse_requests = AnnSearchRequest(the same as above) — the request<br>
results = collection.hybrid_search([dense, sparse_requests], rerank, limit=10, output_fields)<br>
&emsp;&emsp;rerank = RRFRanker() with k=60, or WeightedRanker(0.5, 0.5) — re-ranking by weighted similarity scores<br>
Or concatenate the question and the result, and re-rank by the attention CLS score — a client feature, a model at a local path, reranker(query, documents)<br>
<br>
for ... in results — the list of results for each query<br>
&emsp;&emsp;results[i] — the top-k list<br>
&emsp;&emsp;results[i][j] — the (i+1)-th result<br>
<br>
Concatenating the results, with the citation marker [1]<br>
res = res + f'[{number}] {result text}\n'<br>
<br>
prompt = f'...{system prompt}...{formatted_references}...{query}...'<br>
Give it to the large model<br>
<br>
Encapsulation and modularization

### Agent

Server tools, or local<br>
Memory

ReAct: prompt-based, explicitly showing reasoning (thought) + action + observation<br>
Plan - execute - reflect<br>
Autonomous loop / AutoGPT: compares the gap between the result and the goal, and dynamically generates new subtasks

Calling different functions according to the input: routing, intent recognition<br>
&emsp;&emsp;function call: built in by default, with arguments<br>
&emsp;&emsp;Keyword semantic routing: traditional intent recognition<br>
&emsp;&emsp;Judged by a small LLM, with a well-written system prompt specifying the output; similar to one of the flows of function call

openai-agents library<br>
&emsp;&emsp;Agent, Runner with the responses protocol<br>
&emsp;&emsp;Model compatibility / through the OpenAI API and the OpenAIModel function<br>
&emsp;&emsp;For tools you do not have to write the schema<br>
&emsp;&emsp;as_tool, function_tool, MCP, inner

**Implementation**

The plan module<br>
system prompt1<br>
&emsp;&emsp;cot: analyze the question, decide which queries to use based on the information already available in order to get more of the information needed, and break the question down or extend it to obtain comprehensive information<br>
&emsp;&emsp;The available tools<br>
&emsp;&emsp;Output format, examples: multiple subtasks with tool names<br>
system prompt1 + memory + query = prompt1<br>
The list of subtasks and tool names = model(prompt1)<br>
<br>
The execution module<br>
if tool_name<br>
&emsp;&emsp;result = rag or websearch(subtask prompt)<br>
&emsp;&emsp;Add to memory<br>
<br>
Summarize and output<br>
Add the memory message<br>
system prompt2 + memory + query = prompt2<br>
<br>
Encapsulation<br>
<br>
Real memory has to consider refresh, summarization or conversion<br>
If it is mcp / an external tool call, state the requirements — the prompt on the large model side<br>
<br>
The reflection module<br>
Evaluate the quality after the execution results are returned<br>
system prompt3<br>
If it does not pass, reason it out, with multiple subtasks; if it passes, do nothing and return empty, then summarize and output

**MCP**

MCP does not specify the interaction with the large model / the interaction between the host and the large model<br>
It is the format required by the large model api, with messages<br>
The host relays the messages, carrying role system with the detailed documentation, the tool list, and the output requirements<br>
role user: the question and the information about the files in the current environment<br>
react

The MCP host and server establish a handshake, two-way communication, the available tools and their descriptions: the function names / arguments / docstrings extracted by mcp.tool()<br>
A server includes multiple tools<br>
<br>
Setting up the large model api in Cline<br>
Add or configure an MCP server: the large model chat, the marketplace, configure in json format<br>
transport type: stdio / streaming output, streamable http<br>
command: uvx for the official ones online / uv for a local project / npx / python, etc.<br>
&emsp;&emsp;Cline may not know the path; use which uv / python to get the absolute path<br>
args: the run arguments / a .py local project, or online mcp.so, mcpmarket.com, etc.<br>
<br>
Defining the server and the tools<br>
Install uv; uv add for the mcp dependencies, cli, httpx; and uv pip install<br>
mcp = mcp.server.fastmcp<br>
Some definitions<br>
&emsp;&emsp;The target url of the request, the user's identifier, the request-receiving function and the formatting function, etc.<br>
mcp.tool() def with a docstring<br>
<br>
mcp.run() starts the server<br>
<br>
Defining the host<br>
Import mcp.client.session, .streamable_http, .stdio<br>
&emsp;&emsp;Define a class to integrate the LLM and the host<br>
&emsp;&emsp;&emsp;&emsp;Initialize the model api<br>
&emsp;&emsp;&emsp;&emsp;async def functions<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;server_params with args=[mcp_server.py]<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;x = session.list_tools() with client and server_params<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;for ... in x.tools: the openai tool dict template with x.name, x.description, parameters<br>
&emsp;&emsp;&emsp;&emsp;def a system prompt function<br>
&emsp;&emsp;&emsp;&emsp;async def tool call<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;await session.call_tool(tool name and arguments)<br>
&emsp;&emsp;&emsp;&emsp;def message<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;openai api<br>
The official MCP .client only handles communication; the model and the logic have to be written by yourself<br>
langchain-mcp has already wrapped it all up<br>
<br>
Choosing tools intelligently: the docstrings of the tool definitions, the system prompt, and fine-tuning again on function call<br>

**A2A**

The agent is started as an http service; with http + json the language can be ignored<br>
&emsp;&emsp;url<br>
/.well-known/agent.json: the agent card — the input and output schemas, name, description, skills, url<br>
The jsonrpc request and the response correspond by id<br>
config, message / context, artifact, history / context<br>
The user, the platform sending and receiving on their behalf, the scheduling agent, the weather agent<br>
&emsp;&emsp;A2A when adding / registering: add the agent url, the platform requests the weather agent and gets the card back, and afterwards it is between the platform and the scheduling agent<br>
&emsp;&emsp;A2A when in use: the scheduling agent and the weather agent

**Agent skill**

Skill encapsulation<br>
The name and the function description, input parameters validated with a json schema, the execution logic, structured output with status codes and bug information<br>
&emsp;&emsp;The various tools (functions, APIs, web access, RAG, etc.), examples and prompts, safe handling of errors and failures<br>
<br>
The list of skill metadata is given to the large model, which selects a skill; the skill.md is then given to the large model<br>
Skills are nested progressively: the prompt describes in which scenarios to use the other .md, .py, etc.

**Context and memory**

Layered memory, context engineering<br>
<br>
Large models are stateless and do not add information on their own; context length limits; irrelevant information; token cost<br>
<br>
Context to memory, memory to context<br>
&emsp;&emsp;Current, day-level memory, history<br>
&emsp;&emsp;Core memory as static context; skill memory / workflow file directories loaded progressively<br>
&emsp;&emsp;RAG as dynamic context<br>
Summarizing and compressing the context<br>
Continuous prefix caching

**Verified reward and Agentic RL**

ground truth / objective results / verifiable / with the correctness of the result there is no need to train a reward model — rule-based rewards<br>
&emsp;&emsp;Math, code, whether the test cases pass, whether the API call returns a response, whether the task is finally completed, environment state verification, etc.<br>
RLVR, RLHF<br>
<br>
RLHF: text generation conforms, a single sequence<br>
agentic RL: agent abilities / tools and results, multiple sequences<br>
Cold start: train up to the passing line, using the existing SFT data<br>
&emsp;&emsp;[System Prompt, User Task, Thought_1-n, Action_1-n, Obs_1-n, final_answer]<br>
&emsp;&emsp;Train on the intermediate process and the final answer; the observation is returned by the environment rather than output by the model, so it is usually not included in the loss<br>
&emsp;&emsp;Data mixture: successful data 60-70, error-correction data 15-20, general conversation data 5-10<br>
GRPO on interaction data<br>
&emsp;&emsp;Trajectories over multiple sequences; rule-based milestone rewards on the intermediate results<br>
<br>
&emsp;&emsp;&emsp;&emsp;llm-as-judge is slower and more expensive, generally used offline

**Self-evoluation**

Automatic improvement, so that the execution gets better<br>
&emsp;&emsp;L1: the memory / experience layer, the memory skill<br>
&emsp;&emsp;L2: the system prompt<br>
&emsp;&emsp;L3: the code / functions / configurations themselves, the model parameters<br>
<br>
Observing the full task trajectory and the metrics<br>
Evaluation: scoring with verifiable signals<br>
Analyzing the failures<br>
Generating improvement candidates: memory / skill / prompt<br>
Held-out regression testing: accepted only when it meets the bar, above the baseline / a 95% completion rate<br>
Save + log<br>
<br>
Failures are always written into memory; successes are written selectively / a high score and many steps<br>
&emsp;&emsp;Deduplication, decay / periodic cleanup of the low-frequency and low-success-rate ones, organizing and merging<br>
The failure-analysis reflection prompt: written comprehensively and specifically<br>
&emsp;&emsp;Pass in the task, the trace, the final result, the evaluator output<br>
&emsp;&emsp;Analyze the deviations, causes and examples, reusable rules and anti-rules<br>
<br>
Trigger conditions<br>
Every conversation, after a session / task ends, after a long time without conversation, triggered by a failure or error, periodically every 50 tasks / organized at regular intervals

**Multi-agent**

You can use one model, or multiple models<br>
<br>
LangGraph: a directed-graph state machine, explicit, with conditional branches, loops, and parallelism; the operations are the edges and nodes of the graph; evaluation with LangSmith<br>
Microsoft AutoGen: conversation between agents, implicit / driven by the prompt and the history<br>
CrewAI: already encapsulated; define the roles, tasks, and processes, and the framework handles the rest automatically<br>
Alibaba AgentScope: distributed actor agents and a message bus<br>
OpenAI Swarm: an agent is a function, returning a function<br>
MetaGPT: the standardized procedures of a software company written into the framework<br>
<br>
Multi-agent interaction and communication: sending messages (AutoGen conversations), shared state (the LangGraph state graph), function calls / encapsulated tool calls, event-driven<br>
Multiple agents in the same framework have a native communication mechanism<br>
When agents involve different frameworks, the A2A protocol works between agents of any frameworks

**Architecture** / topology: orchestrator-worker, pipeline, swarm, hierarchical, blackboard<br>
**Paradigms**: react, planning, reflection<br>
**Communication mechanisms, control strategies**: fork and parallelism, routing, interaction and communication, feedback / different from active reflection, supervisor, voting<br>
**Frameworks** / used to build agents: LangGraph, AutoGen, CrewAI, AgentScope, MetaGPT<br>
&emsp;&emsp;workflow.add_edge()

The simplest: a planner, a generator, and an evaluator collaborating with multiple rounds of evaluation

agents.md, the directory structure, progressively, the various .md files<br>
Components or files other than .md

**LangChain / LangGraph**

LangChain for real-world applications; langchain-core for the core abstract definitions<br>
&emsp;&emsp;model components<br>
&emsp;&emsp;prompt engineering components: prompt templates, output parsing<br>
&emsp;&emsp;chains: pipelines connecting the components / programming with LCEL, the LangChain Expression Language: prompt | model | output_parser<br>
&emsp;&emsp;memory components: conversation history, summary memory, and other modes<br>
&emsp;&emsp;retrievers components: RAG<br>
&emsp;&emsp;agents components: tool calling, execution order<br>
LangGraph state graph<br>
&emsp;&emsp;node, edge, state<br>
&emsp;&emsp;checkpoint<br>
&emsp;&emsp;loops<br>
&emsp;&emsp;multi-agent collaboration / a planning agent and an execution agent<br>
&emsp;&emsp;state persistence / time travel<br>
LangSmith<br>
&emsp;&emsp;tracing: records the complete steps, the full chain<br>
&emsp;&emsp;monitoring: request volume, token consumption, error rate<br>
&emsp;&emsp;evaluation: datasets + LLM judge + comparing the old and new versions<br>
&emsp;&emsp;prompt management<br>
LangServe: deployment, providing an api<br>
LangGraph Studio: web-based visualization, started from the CLI<br>
<br>
Each package is installed separately with pip

**Harness**

Beyond the model<br>
Constraints, logging, validation, state recovery<br>
&emsp;&emsp;Tools: argument validation, retry on failure<br>
&emsp;&emsp;Context / memory management<br>
&emsp;&emsp;The workflow<br>
&emsp;&emsp;Trajectories, state, cost, failures<br>
&emsp;&emsp;Security<br>
&emsp;&emsp;sandbox, permissions, audit and oversight<br>
<br>
The requirements list<br>
Complete only one requirement at a time<br>
<br>
Environments for evaluating a model or an agent: lm-eval, OpenCompass, SWE-bench harness, AgentBench<br>
&emsp;&emsp;The lm-eval and OpenCompass environments have multiple dataset benchmarks<br>
&emsp;&emsp;SWE-bench harness is an environment-interaction benchmark; SWE-bench provides the environment<br>
&emsp;&emsp;The AgentBench environment has multiple environment-interaction benchmarks, and the benchmarks themselves are environments<br>
<br>
rubric: the task is completed and the result is correct; the intermediate process and results; efficiency / number of steps / token consumption / time; the accuracy of recognition and invocation

**Deepseek harness**

turn, step, loop, and scheduling<br>
&emsp;&emsp;Cancellation, jumping the queue, the file system, shell, permissions, persistence<br>
Context<br>
Tool execution<br>
&emsp;&emsp;Allow / deny / ask before execution, sandbox, timeout, checking the result<br>
An append-only event log<br>
&emsp;&emsp;Which turn, which step requested the model, the model configuration, the tool arguments and results, recovery / fork branches<br>
Security<br>
Extensibility<br>
&emsp;&emsp;Extending with new capabilities without affecting the core<br>
Multi-agent design

**IDE**

Information automatically added to the system prompt every time<br>
&emsp;&emsp;System environment information, the project root directory, the file tree structure, the currently active file, the contents of the must-read files, etc.<br>
More messages in the flow; the system prompt is added every round, which differs from function call<br>
&emsp;&emsp;system prompt - user - assistant, the model generates or calls - tool result - generates again - system prompt - user - assistant<br>
Dynamic perception: actively calling functions/tools and getting the results back, either judged by the large model or event-driven<br>
&emsp;&emsp;Not at the start of a Q&A; actively call to read the system environment information, the project root directory, the file tree structure<br>
&emsp;&emsp;Read a certain file<br>
&emsp;&emsp;Run a file and return the result<br>
The whole project is not fed into the large model; the IDE runs RAG embedding or an abstract syntax tree over the project<br>
The diff view before and after the change<br>
&emsp;&emsp;The user's approval or rejection is returned to the large model as an execution result, similar to a function / tool call

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

r'' does not interpret escape characters<br>
await can only be used in async def functions, async with/for, cannot appear in normal functions, with/for<br>
&emsp;&emsp;When calling asynchronous functions or methods, await must be used<br>
<br>
When N(0,1), to prevent division by zero, add an epsilon to the denominator<br>
<br>
Move the project to another location to run, manually connect to network or package everything with docker<br>
If only interface/api is needed, use fastapi<br>
<br>
Relative paths in the project<br>
.gitignore file<br>
&emsp;&emsp;xxx/.venv/, xxx/_ pycache _/, *.pyc suffix, *.log<br>
pip freeze > requirements.txt<br>
uv integrated to replace multiple package managers / install pip install / project management pdm pyproject.toml / virtual environment venv / run scripts<br>
The project folder is independent of the virtual environment, but it is best to place it under the project folder and bind it with the project<br>
uv sync automatically reads pyproject.toml, creates a virtual environment, and installs all dependencies<br>
uv run

**qwen model**

qwen2.5-0.5/1.5/3/7B -14/32B -72B llama3-8B -70B dense -instruct sft+rl、deepseek v3 671B MoE<br>
&emsp;&emsp;deepseek-r1-distill-qwen-32B -llama-70B<br>
qwen3-0.6/1.7/4/8B -14/32B -30B-A3B -235B-A22B -instruct<br>
&emsp;&emsp;-coder-480B-A35B-instruct<br>
&emsp;&emsp;-embedding-0.6/4/8B<br>
&emsp;&emsp;-VL-2/4/8/32B-instruct -30B-A3B-instruct -235B-A22B-instruct<br>
&emsp;&emsp;-omni-30B-A3B<br>
&emsp;&emsp;-max close source<br>
qwen3.5-0.8/2/4/9B -35B-A3B -122B-A10B -397B-A17B -instruct、qwen3.8-27B

**FastAPI**

&emsp;&emsp;User tables, usage statistics, automatic switching to a backup model, and other product-level functions<br>
&emsp;&emsp;Natively asynchronous<br>
<br>
pip packages to install<br>
fastapi; the web application server uvicorn[standard]<br>
pydantic<br>
pytest for testing<br>
httpx: the modern successor to requests<br>
Web development<br>
&emsp;&emsp;python-multipart for handling file uploads and form data, passlib[bcrypt] for password hashing, PyJWT / authlib / fastapi-jwt-auth<br>
<br>
AsyncOpenAI<br>
<br>
app = FastAPI(), optional arguments title, description, version<br>
class b(pydantic.BaseModel):<br>
&emsp;&emsp;text: str — the annotated type is validated by pydantic; a value may or may not be assigned; instantiation requires a value<br>
@app.get('/' or '/abc') is read-only; @app.post is for creating, submitting, write operations, and the request body<br>
async def f0(x):<br>
&emsp;&emsp;Operate on x<br>
&emsp;&emsp;await to call a function<br>
&emsp;&emsp;Using AsyncOpenAI, etc.<br>
&emsp;&emsp;return a, return {'text': x.text}<br>
<br>
uvicorn.run('main:app', host, port, reload), or fastapi dev, or uvicorn main:app --reload<br>
&emsp;&emsp;The server terminal<br>
&emsp;&emsp;Going to localhost:8000 shows the returned part<br>
&emsp;&emsp;localhost:8000/docs: the generated api documentation<br>
The client terminal: a request with curl https localhost:8000/abc<br>
<br>
Passing parameters via the url<br>
&emsp;&emsp;url/1: @app.get('/{item_id}') async def f3(item_id)<br>
&emsp;&emsp;The key-value params in Postman, url/?a=1&b=2: @app.get('/') async def f3(a, b)<br>
Passing parameters in the request body<br>
&emsp;&emsp;response = httpx.post(url, json), or on the command line curl \ \<br>
<br>
router = APIRouter() for grouped development<br>
@router.get<br>
app.include_router<br>
<br>
&emsp;&emsp;app.put to update parameters, app.delete<br>
&emsp;&emsp;app.middleware('http'): unified middleware for requests — logging, authentication, rate limiting, etc.<br>
<br>
Testing urls with the Postman software

**Redis** remote dictionary derver<br>
&emsp;&emsp;An in-memory key-value database with fast response and high-speed caching<br>
Stores context and session states

**multilingual**

SFT with Chinese and English data<br>
&emsp;&emsp;A small amount, <1 / 3 / 5%, of cross-language mixed samples<br>
&emsp;&emsp;At inference, the system prompt keeps proper nouns or terms with bilingual annotations<br>
RAG retrieval with Chinese and English data<br>
&emsp;&emsp;A multilingual embedding model<br>
&emsp;&emsp;The second-best alternative: the system prompt translates Chinese/English into English/Chinese; retrieving separately in each language and merging the results is better than concatenated retrieval<br>
<br>
Cross-language ability depends on the large model

**PageIndex** vector-free reasoning RAG

A tree index / metadata with the structure, hierarchy, and content summaries; the large model reasons over the retrieval to locate the pages, maps them back to the original text, and generates the answer from the original text and the question<br>
&emsp;&emsp;No chunking, embedding, or vector retrieval needed<br>
<br>
PageIndexClient() — the cloud, or the model api<br>
&emsp;&emsp;pdf, storage_path<br>
&emsp;&emsp;get_tree summary text, get_page_content<br>
pageindex on github<br>
&emsp;&emsp;page_index_md.py, md_to_tree

**Mineru** gpu cli<br>
txt mode does not need a language to be specified, ocr mode does; the default auto chooses txt/ocr automatically<br>
pipeline, hybrid-medium/high, vlm

**Engineering**

RAG, prompt, simple fine-tuning

Batch tasks, continuous batch processing, asynchronous<br>
Response lag, prefill TTFT, decode TPOT/ITL, inference optimization<br>
Prompt cache

Cost

Failure, paradigm, mechanism and strategy, workflow<br>
&emsp;&emsp;Planning/intent recognition, tool calling, communication, context loss, loop, content deviation/hallucination/task completion<br>
API timeout/JSON format validation/fault tolerance, context management, state persistence, step-by-step reflection and verification, degradation, loop count, completion verification

Harness<br>
Multi-agent, division of labor, communication<br>
Evaluation, quantitative metrics<br>
&emsp;&emsp;Rubrics<br>
Self-evolution, bonus items<br>
&emsp;&emsp;Bad case

Hot context: currently focused content, the latest tool return, added to prompt<br>
Structured task state: state fields stored in database replace putting into context<br>
Cold data archiving: trace, failure records, not included in prompt by default<br>
&emsp;&emsp;Trace, parameter passing, log troubleshooting<br>
&emsp;&emsp;Filter by metadata, retrieve on demand<br>
Loop detector for dead loop detection, 90s asynchronous timeout, exception converted to tool string for fault tolerance

**Project**

datapipeline

model, device

SFT, inference

index

retrieve

agent flow

input/output optimization







