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
    past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None


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
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=True)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias = True)
        # implement a dropout layer. Zero-out p percentage of values, help with regularization
        # no effect during forward pass.
        self.dropout = nn.Dropout(p = 0.1)
        mask = torch.tril(torch.ones(config.max_ctx_len, config.max_ctx_len)) # create a low_triangular matrix with 1s
        self.register_buffer('bias', mask.view(1, 1, config.max_ctx_len, config.max_ctx_len))
        # and update forward accordingly
    def forward(
            self,
            hidden_states: Tensor,
            past_key_values: Optional[Tuple[Tensor, Tensor]] = None
                ):
        B, T, _ = hidden_states.shape  # (batch_size, Seq_length)
        qkv = self.c_attn(hidden_states)
        Q, K, V = qkv.split(self.config.d_model, dim=-1)
        Q = Q.view(B, T, self.config.n_head, self.config.d_head).transpose(1, 2)
        K = K.view(B, T, self.config.n_head, self.config.d_head).transpose(1, 2)
        V = V.view(B, T, self.config.n_head, self.config.d_head).transpose(1, 2)
        if past_key_values is not None:
            K = torch.cat([past_key_values[0],K], dim = 2)
            V = torch.cat([past_key_values[1],V], dim = 2)
        T_full = K.shape[2]
        Q_len = Q.shape[2]
        scores = Q @ K.transpose(-2,-1) / math.sqrt(self.config.d_head)
        q_start = T_full - Q_len
        mask = self.bias[0, 0, q_start:q_start + Q_len, :T_full] # only need the sequence length for the mask
        scores = scores.masked_fill(mask == 0, float('-inf')) # mask the scores
        attn_weights = torch.softmax(scores, dim=-1)
        values = attn_weights @ V
        values = self.dropout(values)
        values = values.transpose(1, 2)  # → (batch, seq_len, n_heads, d_head)
        values = values.contiguous().view(B, -1, self.config.d_model)  # → (batch, seq_len, d_model)
        return self.c_proj(values), (K, V)
class Multi_Layer_Perceptron(nn.Module):
    def __init__(self, config: GPT2Config = GPT2Config()):
        super().__init__()
        self.config = config
        self.c_fc = nn.Linear(self.config.d_model, self.config.d_mlp_intermediate)
        self.c_proj = nn.Linear(self.config.d_mlp_intermediate, self.config.d_model)
        self.dropout = nn.Dropout(p = 0.1)
    def forward(
            self,
            x: Tensor):
        intermediate = self.c_fc(x)
        act = F.gelu(intermediate,approximate = 'tanh')
        return self.dropout(self.c_proj(act))
class GPT2TransformerBlock(nn.Module):
    def __init__(self, config: GPT2Config = GPT2Config()):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(p = 0.1)
        self.attn = MultiHeadAttention(config)
        self.mlp = Multi_Layer_Perceptron(config)
    def forward(self,
                x: Tensor,
                past_key_values : Optional[Tuple[Tensor, Tensor]] = None ):
        attn_out, new_kv = self.attn(self.ln_1(x), past_key_values)
        x = x + self.dropout(attn_out)
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv
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
        self.wte = nn.Embedding(self.config.vocab_size,self.config.d_model)
        self.wpe = nn.Embedding(self.config.max_ctx_len, self.config.d_model)
        self.ln_f = nn.LayerNorm(self.config.d_model)
        self.h = nn.ModuleList([GPT2TransformerBlock(config) for _ in range(config.n_layer)])
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        #load weights if bin_path is provided.
        if bin_path:
            checkpoint = torch.load(bin_path)
            transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                          'mlp.c_fc.weight', 'mlp.c_proj.weight']

            checkpoint = {
                k: v.t() if any(k.endswith(t) for t in transposed) else v
                for k, v in checkpoint.items()
            }
            self.load_state_dict(checkpoint, strict=False)

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
        #input_ids --> (batch_size, seq_len)
        B, T = input_ids.shape
        ctx_embd = self.wte(input_ids) # match input_ids to embeddings.
        past_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        positions = torch.arange(past_len, past_len + T, device=input_ids.device) # create position indices [0, 1, 2, ..., seq_len-1]
        pos_embd = self.wpe(positions)
        inputs = ctx_embd + pos_embd
        new_kv_cache = []
        for i, block in enumerate(self.h):
            old_kv_cache = past_key_values[i] if past_key_values is not None else None
            inputs, new_kv = block(inputs, old_kv_cache)
            new_kv_cache.append(new_kv)
        inputs = self.ln_f(inputs)
        logits = self.lm_head(inputs)
        return CausalLMOutput(logits=logits, past_key_values = new_kv_cache)
        
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
        self.eval()
        with torch.no_grad():
            past_key_values = None
            for step in range(max_new_tokens):
                ids_to_pass = input_ids if past_key_values is None else next_token
                output = self.forward(ids_to_pass, past_key_values)
                past_key_values = output.past_key_values
                logits = output.logits #(batch, seq_length, d_vocab)
                last_tkn = logits[:, -1,:]
                if temperature == 0.0:
                    next_token = torch.argmax(last_tkn, dim=-1, keepdim=True)
                else:
                    probs = torch.softmax(last_tkn / temperature, dim=-1)
                #nucleus sampling here.
                    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                    cumSum = torch.cumsum(sorted_probs, dim=-1)
                    sorted_probs[cumSum > top_p] = 0  # zero where cumsum > top_p
                    sorted_probs[:, 0] = sorted_probs[:, 0].clamp(min=1e-9)  # always keep top
                    sampled = torch.multinomial(sorted_probs, num_samples = 1)
                    next_token = sorted_indices.gather(-1, sampled)
                input_ids = torch.cat([input_ids, next_token], dim=1)
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
        self.config = config
        self.base_model = GPT2LMHeadModel(config) # reuse the base_model developped earlier
        self.classifier = nn.Linear(self.config.d_model, self.config.num_labels)
        if lm_bin_path is not None:
            self.base_model = GPT2LMHeadModel(config, bin_path=lm_bin_path)
        if classifier_bin_path is not None:
            self.load_state_dict(torch.load(classifier_bin_path))
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
        x = self.base_model.wte(input_ids) + self.base_model.wpe(torch.arange(input_ids.shape[1], device=input_ids.device)) # hidden_states
        # pass through all transformer block
        for block in self.base_model.h:
            x, _ = block(x)
        x = self.base_model.ln_f(x)
        last_token = x[:, -1, :] # take the embeddings of the last token
        logits = self.classifier(last_token)
        return SequenceClassifierOutput(logits=logits)

