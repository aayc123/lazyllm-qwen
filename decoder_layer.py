from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

import torch
from context import Context
import torch.nn as nn
from typing import Tuple, Optional
from config import LazyQwen3Config
decode_step_count = [0]

class DecoderLayer(nn.Module):
    """
    A custom decoder layer that builds upon the Qwen3DecoderLayer and implements dynamic token pruning.

    This layer utilizes KV cache and Aux Cache to speed up the "time-to-first-token" (TTFT) of Hugging 
    Face's Qwen3 implementation. Dynamic token pruning is used in each forward pass, based on attention 
    importance scores and pruning rates defined in the configuration. 
    """
    def __init__(self, config: LazyQwen3Config, layer_idx: int):
        """
        Initializes the decoder layer.

        Args:
            config (LazyLlamaConfig): Configuration object containing model hyperparameters.
            layer_idx (int): The index of the current decoder layer.
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
       
        sdpa_config = config.__class__(**config.to_dict())
        sdpa_config._attn_implementation = "sdpa"
        self.decoder = Qwen3DecoderLayer(sdpa_config, 0)
        # The Qwen3DecoderLayer needs the layer index to retrieve the correct KV Cache, however we only pass it the 
        # KV Cache of the current layer. Therefore the layer index needs to be 0 for all Qwen3DecoderLayers.
        # self.decoder = Qwen3DecoderLayer(config, 0)
        # print(f"[DecoderLayer] attn_implementation: {self.decoder.self_attn.config._attn_implementation}")

    def forward(
            self,
            context: Context,
            causal_mask: torch.FloatTensor,
            rotary_emb: Qwen3RotaryEmbedding,
            output_attentions: bool,
        ) -> Tuple[Context, Optional[torch.Tensor]]:
        """
        Executes the forward pass for the decoder layer, updating the hidden states and caches, 
        and optionally returning attention weights.

        Args:
            context (Context): The context object containing hidden states, KV Cache, Aux Cache, etc.
            causal_mask (torch.Tensor): The 4D causal mask for the attention mechanism. 
            rotary_emb (LlamaRotaryEmbedding): The rotary embedding layer. This must be passed to the decoder since it 
                recomputes the embeddings for tokens from the Aux Cache.
            output_attentions (bool): Whether to return attention weights.
        """
        local_kv_cache = context.get_kv_cache(self.layer_idx)
        # if self.layer_idx == 0 and context.hidden_states.shape[1] == 1:
        #     print(f"[decode layer0] in_kv_cache_idxs len={context.in_kv_cache_idxs.shape[0]}")
        #     print(f"[decode layer0] cache_status sum={context.kv_cache.cache_status_bit_array[0].sum().item()}")
        #     print(f"[decode layer0] selected_tokens sum={context.selected_tokens_bit_array.sum().item()}")
        #     # 检查KV值本身是否有效
        #     kv_sample = context.kv_cache.key_cache[0][0, 0, :5, :3]
        #     print(f"[decode layer0] key_cache[0][0,:5,:3]={kv_sample}")

        if self.layer_idx > 0:
            context.get_aux_cache(self.layer_idx)
        if self.layer_idx == 0 and context.hidden_states.shape[1] > 1:
            hs = context.hidden_states
            print(f"[layer0 input] mean={hs.mean().item():.4f}, std={hs.std().item():.4f}, max={hs.max().item():.4f}")
        
        causal_mask = torch.index_select(causal_mask, 3, context.keys_idxs_to_tokens_idxs)
        causal_mask = torch.index_select(causal_mask, 2, context.hidden_states_idxs)
        causal_mask = causal_mask.to(context.hidden_states.device)
        
        if self.layer_idx == 0 and context.hidden_states.shape[1] > 1:
            pos = context.hidden_states_positions
            print(f"[layer0 positions] shape={pos.shape}, first5={pos[0,:5].tolist()}, last5={pos[0,-5:].tolist()}")
        position_embeddings = rotary_emb(context.hidden_states, context.hidden_states_positions)
        
        residual = context.hidden_states
        with torch.no_grad():
            hidden_states_norm = self.decoder.input_layernorm(context.hidden_states)
        kv_cache_len_before = context.in_kv_cache_idxs.shape[0]

        is_prefill = context.hidden_states.shape[1] > 1
        # causal_mask = causal_mask.float()
        if self.layer_idx == 0 and context.hidden_states.shape[1] > 1:
            print(f"[mask values] min={causal_mask.min():.1f}, max={causal_mask.max():.1f}")
            print(f"[mask dtype] {causal_mask.dtype}")
            print(f"[hidden_states_norm] min={hidden_states_norm.min():.4f}, max={hidden_states_norm.max():.4f}, has_nan={hidden_states_norm.isnan().any()}")
        if context.hidden_states.shape[1] > 1 and self.layer_idx <= 2:
            print(f"[layer{self.layer_idx} pre-attn] hidden_states.shape={context.hidden_states.shape}")
            print(f"[layer{self.layer_idx} pre-attn] hidden_states_idxs: first5={context.hidden_states_idxs[:5].tolist()}, last5={context.hidden_states_idxs[-5:].tolist()}")
            print(f"[layer{self.layer_idx} pre-attn] in_kv_cache_idxs.shape={context.in_kv_cache_idxs.shape}")
            print(f"[layer{self.layer_idx} pre-attn] causal_mask.shape={causal_mask.shape}")
        with torch.no_grad():
            hidden_states, _ = self.decoder.self_attn(
                hidden_states=hidden_states_norm,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                past_key_values=local_kv_cache,
                cache_position=context.hidden_states_idxs,
                output_attentions=False,
            )
            hidden_states = hidden_states.to(context.hidden_states.dtype)
            if self.layer_idx <= 2 and context.hidden_states.shape[1] > 1:
                print(f"[layer{self.layer_idx}] attn_out.shape={hidden_states.shape}")
                print(f"[layer{self.layer_idx}] attn_out: mean={hidden_states.mean():.4f}, std={hidden_states.std():.4f}, max={hidden_states.abs().max():.4f}")
        if context.hidden_states.shape[1] > 1:
            if hidden_states.isnan().any():
                print(f"[NaN] layer={self.layer_idx}, NaN appeared after self_attn")
            elif (residual + hidden_states).isnan().any():
                print(f"[NaN] layer={self.layer_idx}, NaN appeared after residual+hidden")
        # ✅ prefill 阶段用 hook 捕获的 Q/K 计算重要性分数
        if is_prefill:
            with torch.no_grad():
                q = self.decoder.self_attn.q_proj(hidden_states_norm)
                k = self.decoder.self_attn.k_proj(hidden_states_norm)
                batch, seq_len, _ = q.shape
                q = q.view(batch, seq_len, self.config.num_attention_heads, self.config.head_dim).transpose(1, 2)
                k = k.view(batch, seq_len, self.config.num_key_value_heads, self.config.head_dim).transpose(1, 2)
                cos, sin = position_embeddings
                from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                k = repeat_kv(k, self.config.num_attention_heads // self.config.num_key_value_heads)
                last_token_query_idx = context.tkns_idxs_to_hidden_states_idxs[context.sequence_length - 1]
                q_last = q[:, :, last_token_query_idx:last_token_query_idx+1, :]
                scale = self.config.head_dim ** -0.5
                attn_scores = torch.matmul(q_last, k.transpose(-2, -1)) * scale
                attn_scores = torch.softmax(attn_scores, dim=-1)
                importance_scores_list = attn_scores[0, :, 0, :].mean(dim=0).detach().clone()
                del q, k, q_last, attn_scores, cos, sin

        with torch.no_grad():
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.decoder.post_attention_layernorm(hidden_states)
            hidden_states = self.decoder.mlp(hidden_states)
            hidden_states = residual + hidden_states
        if self.layer_idx == 0 and context.hidden_states.shape[1] > 1:
            print(f"[layer0 output] mean={hidden_states.mean().item():.4f}, std={hidden_states.std().item():.4f}, max={hidden_states.max().item():.4f}")
        if self.layer_idx == 35 and context.hidden_states.shape[1] > 1:
            print(f"[layer35 output] mean={hidden_states.mean().item():.4f}, std={hidden_states.std().item():.4f}, max={hidden_states.max().item():.4f}")
            
        if context.hidden_states.shape[1] > 1 and hidden_states.isnan().any():
            print(f"[NaN] layer={self.layer_idx}, NaN appeared after MLP")
        context.hidden_states = hidden_states
        context.update_kv_cache(local_kv_cache, self.layer_idx, kv_cache_offset=kv_cache_len_before)

        # 剪枝
        if is_prefill:
            last_token_key_idx =  last_token_query_idx
            pruning_rate = self.config.pruning_rates[self.layer_idx]
            if importance_scores_list.shape[0] > 1:
                importance_scores_list[last_token_key_idx] = float("inf")
                _, to_prune_list_idxs = torch.topk(
                    importance_scores_list,
                    int(pruning_rate * importance_scores_list.shape[0]),
                    largest=False
                )
                to_prune_list_idxs = to_prune_list_idxs + kv_cache_len_before  
            else:
                to_prune_list_idxs = torch.tensor([], dtype=torch.long, device=context.device)
        else:
            to_prune_list_idxs = torch.tensor([], dtype=torch.long, device=context.device)

        to_prune_idxs = context.keys_idxs_to_tokens_idxs[to_prune_list_idxs]

        if self.layer_idx < self.config.num_hidden_layers - 1:
            context.update_aux_cache(to_prune_idxs, self.layer_idx)

        context.prune(to_prune_idxs)
        if is_prefill:
            del importance_scores_list
        outputs = (context,)
        return outputs