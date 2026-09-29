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
    Tokenizer 123...1000... Length information but no dimensional attributes  
        Dense information, inability to handle polysemy, multi-word combinations  
    One-hot (1,0,0) Dimensional attributes but no length information  
        Semantic space, all orthogonal, sparse  
Dimensionality reduction/compression facilitates 1D dimensionality increase/reconstruction of compressed data  
Tokenizer first, then one-hot, followed by semantic space

Matrix multiplication/Vector-matrix operations/Matrix effects determined by matrix/Spatial transformation/Spatial relationships  
/New coordinate system transformation/Linear point-to-point mapping/Dimensionality increase, decrease, rotation, scaling/Invariant inter-vector relationships

Neural networks/Non-linear spatial transformations/Dimension-increasing kernel functions/Hierarchical feature extraction with dimensionality changes  
    Matrix operations/Dimensionality changes, parameters, as above  
    Activation functions/Non-linear/Non-one-to-one correspondence  
Matrix operations/GPU

  Fully connected networks cannot handle polysemy/Tensors parallel to network
  Context required, leading to attention mechanisms

1960s non-parametric attention/Distance kernel regression
  Data (xi,yi) or vector y, query x
  Scalar form f(x)=sum_i alpha(x,xi)*yi, attention weights alpha
Parameterized attention/Introduction of learnable parameters
Score/Similarity/Correlation of q·k, Weights/Softmax of scores
  Score function design Wqk

Structure
Encoder non-autoregressive, Decoder autoregressive, KV cross Q/Translation/Only removed
Input encoding module/Embedding + positional semantics, Feature module, Task requirement module/Linear + Softmax
Attention, Fully connected feedforward 1ReLU2/+ResNet+Norm
Content
  Attention position corresponds to V sequence semantic space influence
  W matrix din=dout, Q matrix K matrix/sqrt(dout)/Vectors in QK are dout-dimensional independent N(0,1)/Product N(0,dout)
  Din dimension split heads/Dout split heads, Feedforward dimensionality changes, Hidden layer d-task linear layer logits

### Machine learning and deep learning

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

