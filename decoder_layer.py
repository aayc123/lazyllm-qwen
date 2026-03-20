from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding
import torch
from context import Context
import torch.nn as nn
from typing import Tuple, Optional
from config import LazyQwen3Config
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

        # The Qwen3DecoderLayer needs the layer index to retrieve the correct KV Cache, however we only pass it the 
        # KV Cache of the current layer. Therefore the layer index needs to be 0 for all Qwen3DecoderLayers.
        self.decoder = Qwen3DecoderLayer(config, 0)

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

        if self.layer_idx > 0:
            context.get_aux_cache(self.layer_idx)

        '''
            LLaMA 2 在 models.py 里用 _prepare_4d_causal_attention_mask_with_cache_position 
            生成一个全局统一的 4D mask。
            Qwen3 引入了 Sliding Window Attention,
            Qwen3DecoderLayer 内部会根据当前层的 layer_type(full_attention 或 sliding_attention)
            自己选择 mask。
            最简单的处理方式：把 causal_mask 从一个 tensor 变成一个 dict,分别存两种 mask
        '''
        causal_mask = torch.index_select(causal_mask, 3, context.keys_idxs_to_tokens_idxs)
        causal_mask = torch.index_select(causal_mask, 2, context.hidden_states_idxs)

        position_embeddings = rotary_emb(context.hidden_states, context.hidden_states_positions)

        residual = context.hidden_states
        hidden_states = self.decoder.input_layernorm(context.hidden_states)

        hidden_states, attention_weights = self.decoder.self_attn(
            hidden_states=hidden_states,
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
            past_key_values=local_kv_cache,
            # cache_position=torch.arange(
            #     context.in_kv_cache_idxs.shape[0],
            #     context.hidden_states.shape[1] + context.in_kv_cache_idxs.shape[0],
            #     device=context.device
            # ),
            cache_position=context.hidden_states_idxs,
            output_attentions=True,   # ← 关键，强制返回attention_weights
        )
        # if self.layer_idx == 0:
        #     print(f"[DEBUG] context.hidden_states.shape: {context.hidden_states.shape}")
        #     print(f"[DEBUG] context.sequence_length: {context.sequence_length}")
        #     print(f"[DEBUG] context.hidden_states_idxs: {context.hidden_states_idxs}")
        
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.decoder.post_attention_layernorm(hidden_states)
        hidden_states = self.decoder.mlp(hidden_states)
        hidden_states = residual + hidden_states

        context.hidden_states = hidden_states
        context.update_kv_cache(local_kv_cache, self.layer_idx)
        
        '''
        取出最后一个 token 对所有其他 token 的 attention 权重，
        在 batch 和 head 维度上取均值，得到每个 token 的重要性分数。
        '''
        last_token_query_idx = context.tkns_idxs_to_hidden_states_idxs[-1]
        # The last token key's index will be the index of the last token in the hidden states, plus the number of tokens in the KV Cache.
        # This is because the KV Cache always comes before the hidden states in the attention mechanism.
        last_token_key_idx = context.in_kv_cache_idxs.shape[0] + last_token_query_idx

        attn_weights_to_last_tkn = attention_weights[:, :, last_token_query_idx, :]
        importance_scores_list = torch.sum(attn_weights_to_last_tkn, dim=(0,1)) / (attention_weights.shape[0] * attention_weights.shape[1])
        '''
        用 torch.topk(..., largest=False) 选出重要性最低的 pruning_rate * N 个 token
        '''
        pruning_rate = self.config.pruning_rates[self.layer_idx]

        if importance_scores_list.shape[0] > 1:
            # Setting the last token's importance to infinity, because we don't want to prune it
            importance_scores_list[last_token_key_idx] = float("inf")
            _, to_prune_list_idxs = torch.topk(importance_scores_list, int(pruning_rate * importance_scores_list.shape[0]), largest=False)
        else:
            to_prune_list_idxs = torch.tensor([], dtype=torch.long, device=context.device)

        to_prune_idxs = context.keys_idxs_to_tokens_idxs[to_prune_list_idxs]

        if self.layer_idx < self.config.num_hidden_layers - 1:
            context.update_aux_cache(to_prune_idxs, self.layer_idx)

        context.prune(to_prune_idxs)
        
        outputs = (context,)
        
        if output_attentions:
            outputs += (attention_weights,)
        if self.layer_idx == 0 and context.hidden_states.shape[1] > 1:
            print(f"[PREFILL] importance_scores_list: {importance_scores_list}")
            print(f"[PREFILL] last_token_query_idx: {last_token_query_idx}")
            print(f"[PREFILL] last_token_key_idx: {last_token_key_idx}")
            print(f"[PREFILL] to_prune_idxs: {to_prune_idxs}")
            print(f"[PREFILL] attention_weights shape: {attention_weights.shape}")
            print(f"[PREFILL] attn_weights_to_last_tkn: {attn_weights_to_last_tkn}")
        return outputs