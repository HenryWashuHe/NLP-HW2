"""
GPT-2 Implementation from Scratch

This module implements the GPT-2 transformer architecture using only PyTorch.
No HuggingFace dependencies are allowed in this file.
"""

import math
from typing import Optional, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class GPT2Config:
    """Configuration class for GPT-2 small model."""
    # Total number of tokens in the vocabulary.
    # Note we use the same vocabulary and tokenizer as OpenAI GPT-2.
    vocab_size: int = 50257
    
    # The maximum context window length is 1024 tokens for GPT-2.
    max_ctx_len: int = 1024
    
    # The model dimension (hidden size) for GPT-2 Small is 768.
    d_model: int = 768
    
    # The dimension of each attention head is d_model / n_head = 768 / 12 = 64.
    d_head: int = 64
    
    # The intermediate dimension of the MLP in GPT-2 Small is 4 times the model dimension.
    # 4 * 768 = 3072
    d_mlp_intermediate: int = 3072
    
    # GPT-2 Small has 12 transformer blocks.
    n_layer: int = 12
    
    # GPT-2 Small has 12 attention heads per transformer block.
    n_head: int = 12
    
    # Total number of label classes for our classification dataset.
    num_labels: int = 20


@dataclass
class CausalLMOutput:
    """Output class for causal language modeling. Contains the logits for all input tokens."""
    logits: Tensor


@dataclass
class ModelOutput:
    """Output class for generation. Contains sequences of input and generated token IDs."""
    sequences: Tensor


@dataclass
class SequenceClassifierOutput:
    """Output class for sequence classification. Contains the logits for each label class."""
    logits: Tensor

class MultiHeadAttention(nn.Module):
    def __init__(self, config : GPT2Config = GPT2Config()):
        super().__init__()
        # W_Q --> d_model x d_model Q = W_Q^T x X = B x T x d_k
        self.config =  config
        #Q, K, V matrix and output matrix that project back to d_model
        self.W_Q = nn.Linear(config.d_model, config.d_model, bias = True)
        self.W_K = nn.Linear(config.d_model, config.d_model, bias = True)
        self.W_V = nn.Linear(config.d_model, config.d_model, bias = True)
        self.W_O = nn.Linear(config.d_model, config.d_model, bias = True)
        # implement a dropout layer. Zero-out p percentage of values, help with regularization
        # no effect during forward pass.
        self.DropOut = nn.Dropout(p = 0.1)
        mask = torch.tril(torch.ones(config.max_ctx_len, config.max_ctx_len)) # create a low_triangular matrix with 1s
        self.register_buffer('mask', mask)
    def forward(
            self,
            hidden_states: Tensor
                ):
        Q = self.W_Q(hidden_states) #(batch, seq_len, d_model)
        K = self.W_K(hidden_states)
        V = self.W_V(hidden_states)
        B, T, _ = hidden_states.shape # (batch_size, Seq_length)
        # split d_model into n_heads and d_head
        Q = Q.view(B, T, self.config.n_head, self.config.d_head).transpose(1,2)
        K = K.view(B, T, self.config.n_head, self.config.d_head).transpose(1,2)
        V = V.view(B, T, self.config.n_head, self.config.d_head).transpose(1,2)
        scores = Q @ K.transpose(-2,-1) / math.sqrt(self.config.d_head)
        mask = self.mask[:T,:T] # only need the sequence length for the mask
        scores = scores.masked_fill(mask == 0, float('-inf')) # mask the scores
        attn_weights = torch.softmax(scores, dim=-1)
        values = attn_weights @ V
        values = self.DropOut(values)
        values = values.transpose(1, 2)  # → (batch, seq_len, n_heads, d_head)
        values = values.contiguous().view(B, T, self.config.d_model)  # → (batch, seq_len, d_model)
        return self.W_O(values)
class Multi_Layer_Perceptron(nn.Module):
    def __init__(self, config: GPT2Config = GPT2Config()):
        super().__init__()
        self.config = config
        self.Linear_1 = nn.Linear(config.d_model, config.d_mlp_intermediate)
        self.Linear_2 = nn.Linear(config.d_mlp_intermediate, config.d_model)
        self.DropOut = nn.Dropout(p = 0.1)
    def forward(
            self,
            x: Tensor):
        intermediate = self.Linear_1(x)
        act = F.gelu(intermediate,approximate = 'tanh')
        return self.DropOut(self.Linear_2(act))
class GPT2TransformerBlock(nn.Module):
    def __init__(self, config: GPT2Config = GPT2Config()):
        super().__init__()
        self.config = config
        self.layerNorm_1 = nn.LayerNorm(config.d_model)
        self.layerNorm_2 = nn.LayerNorm(config.d_model)
        self.DropOut = nn.Dropout(p = 0.1)
        self.attention = MultiHeadAttention(config)
        self.mlp = Multi_Layer_Perceptron(config)
    def forward(self, x: Tensor):
        x = x + self.DropOut(self.attention(self.layerNorm_1(x)))
        x = x + self.mlp(self.layerNorm_2(x))
        return x
class GPT2LMHeadModel(nn.Module):
    """
    GPT-2 Language Model with a language modeling head.
    This corresponds to HF's GPT2LMHeadModel.
    """

    def __init__(self, config: GPT2Config = GPT2Config(), bin_path: Optional[str] = None):
        """
        Initialize GPT-2 Language Model.
        
        Args:
            config: GPT2Config object containing model configurations.
            bin_path: Path to the pytorch_model.bin file. If empty or None, 
                      weights will not be loaded from file.
        """
        super().__init__()
        # TODO: define and initialize the GPT-2 model architecture here.
        # If the `bin_path` argument is provided,
        # load the model weights from the specified file path.
        # If `bin_path` is empty or None, do not load any weights,
        # and initialize the model with random weights.
        self.config = config
        #initliaze context and positionl embeddings with uniform distribution. Dimension |V| x d_model and max_ctx_len x d_model
        self.contextEmbeddings = nn.Embedding(self.config.vocab_size,self.config.d_model)
        self.positionalEmbeddings = nn.Embedding(self.config.max_ctx_len, self.config.d_model)
        nn.init.uniform_(self.contextEmbeddings.weight,a = -0.1, b = 0.1)
        nn.init.uniform_(self.positionalEmbeddings.weight,a = -0.1, b = 0.1) 
        self.layer_norm = nn.LayerNorm(self.config.d_model)
        self.blocks = nn.ModuleList([GPT2TransformerBlock(config) for _ in range(config.n_layer)])
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.contextEmbeddings.weight
        #load weights if bin_path is provided.
        if bin_path:
            self.load_state_dict(torch.load(bin_path))

    def forward(
        self, 
        input_ids: Tensor, 
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
    ) -> CausalLMOutput:
        """
        Forward pass of GPT-2.
        
        Args:
            input_ids: [batch_size, seq_len] token IDs
            past_key_values: Optional list of past key-value pairs for KV caching

        Returns:
            CausalLMOutput with logits
        """
        # TODO: implement the GPT-2 forward pass here. 
        # The forward pass should compute the output logits for all input tokens,
        # and also update the cached attention keys and values in place (reference passing) 
        # if `past_key_values` is provided.

        
        
        return CausalLMOutput(logits=logits)
        
    def generate(
        self,
        input_ids: Tensor,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_new_tokens: int = 128
    ) -> ModelOutput:
        """
        Generate tokens autoregressively using KV caching.
        
        Args:
            input_ids: [batch_size, seq_len] starting token IDs
            temperature: Sampling temperature. If 0.0, use greedy sampling.
            top_p: Top-p (nucleus) sampling threshold
            max_new_tokens: Maximum number of new tokens to generate
        
        Returns:
            ModelOutput with `sequences` containing the generated token IDs
        """        
        # TODO: implement the generation method here. 
        # You should use the `forward` method to compute logits and update KV cache at each step.
        # You can assume the input sequences are always padded to the same length,
        # and the total sequence length (input + generated) will not exceed 512 tokens.
        # GPT-2 does not have a stop token,
        # so you should always generate `max_new_tokens` new tokens 
        # for all the input sequences in the batch.
        
        return ModelOutput(sequences=input_ids)


class GPT2ForSequenceClassification(nn.Module):
    """
    GPT-2 Model with a classification head.
    """

    def __init__(self, 
                 config: GPT2Config = GPT2Config(), 
                 classifier_bin_path: Optional[str] = None,
                 lm_bin_path: Optional[str] = None):
        """
        Initialize GPT-2 Classification Model.
        
        Args:
            config: GPT2Config object containing model configurations,
                    including the number of labels.
            classifier_bin_path: Path to the 
                    This file should contain the weights for 
                    both the GPT-2 base model and the classification head.
                    If empty or None,
                    the classification head weights will be initialized randomly, 
                    and the base model weights may be initialized randomly 
                    or loaded from `lm_bin_path` if provided.
            lm_bin_path: Path to the pytorch_model.bin file for the language model.
                    This file should contain the weights for the GPT-2 base model.
                    If empty or None,
                    weights may be initialized randomly, 
                    or loaded from `classifier_bin_path` if provided.
        """
        super().__init__()

        # Only one of `classifier_bin_path` and `lm_bin_path` can be provided.
        assert not (classifier_bin_path and lm_bin_path), \
            "Only one of `classifier_bin_path` and `lm_bin_path` can be provided."

        # TODO: define and initialize the GPT-2 model that can be used for sequence classification.
        # You can reuse the GPT2LMHeadModel defined above as the base model,
        # and add a classification head on top of it.
        # You should also reuse GPT2LMHeadModel's weights to speed up training if possible.

    def forward(self, input_ids: Tensor) -> SequenceClassifierOutput:
        """
        Forward pass of GPT-2 for classification.
        
        Args:
            input_ids: [batch_size, seq_len] token IDs
        
        Returns:
            SequenceClassifierOutput with logits of shape (batch_size, num_labels)
        """
        
        # TODO: implement the forward pass for sequence classification here.
        # The output logits should be of shape (batch_size, num_labels),
        # where num_labels is specified in the GPT2Config,
        # and the logits contain the classification scores for each label class.
        
        return SequenceClassifierOutput(logits=logits)
