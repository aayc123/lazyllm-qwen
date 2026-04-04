import torch
from caches import KVCache, AuxCache, HFCache
a=0
b=0
class Context:
    """
    This class provides a way to manage the state of tokens inbetween the layers of the transformer model.
    It keeps track of which tokens are in the hidden states, KV Cache, and Aux Cache, and provides methods to 
    update the caches and hidden states, and prune the tokens.
    """
    def __init__(
            self, 
            hidden_states: torch.FloatTensor, 
            kv_cache: KVCache,
            aux_cache: AuxCache,
            tokens_positions_idxs: torch.LongTensor, 
            hidden_states_idxs: torch.LongTensor,
            sequence_length: int 
        ):
        """
        Initializes the Context with the hidden states, KV Cache, Aux Cache, token positions, and hidden states indexes.
        
        Args:
            hidden_states (torch.FloatTensor): The hidden states of the transformer model. 
                The shape is (batch_size, sequence_length, hidden_size).
            kv_cache (KVCache): The KV Cache for the transformer model.
            aux_cache (AuxCache): The Aux Cache for the transformer model.
            tokens_positions_idxs (torch.LongTensor): The positions of the tokens in the sequence. 
                Serves a similar purpose to `position_ids` in the original code, but it's storing the positions of all tokens 
                in the sequence, not just the ones in the hidden states. The shape is (batch_size, sequence_length).
            hidden_states_idxs (torch.LongTensor): The indexes of the hidden states in the sequence. Equivalent to `cache_position`
                in the original code. The shape is (sequence_length).
            sequence_length (int): The current length of the sequence.
        """
        assert hidden_states.shape[1] == 1 or hidden_states.shape[1] == sequence_length, \
            "The sequence length must either match the hidden states or there should be only one token in the hidden states"
        
        self.kv_cache = kv_cache
        self.aux_cache = aux_cache

        self.sequence_length = sequence_length
        self.device = hidden_states.device

        self.hidden_states = hidden_states

        self.tokens_positions_idxs = tokens_positions_idxs

        self.hidden_states_idxs = hidden_states_idxs
        
        max_sequence_length = kv_cache.size[2]

        if int(sequence_length) > int(max_sequence_length):
            raise ValueError(
                f"Context sequence_length={int(sequence_length)} exceeds KVCache capacity={int(max_sequence_length)}. "
                "Increase MAX_SEQ_LEN (prompt+generation) or truncate the prompt to leave generation budget."
            )

        self.selected_tokens_bit_array = torch.zeros(max_sequence_length, device=self.device, dtype=torch.bool)
        self.selected_tokens_bit_array[torch.arange(sequence_length, device=self.device)] = True

        self.in_kv_cache_idxs = None

        self._update_keys_idxs_to_tokens_idxs = True
        self._update_tkns_idxs_to_hidden_states_idxs = True

        # 上一层写入的、按「绝对 token 位置」索引的重要性分数；NaN 表示尚未有分数（下一层预剪枝时视为保留）。
        self.token_importance_prev = torch.full(
            (max_sequence_length,), float("nan"), device=self.device, dtype=torch.float32
        )

    def save_pre_prune_drops_to_aux(self, drop_token_abs: torch.LongTensor, layer_idx: int) -> None:
        """Save hidden states of pre-pruned tokens so get_aux_cache(layer_idx+1) can restore them.

        Writes to aux_cache[layer_idx] (not layer_idx-1), because get_aux_cache(L) reads
        aux_cache[L-1].  So tokens pruned at layer N entrance are found by layer N+1.
        """
        if drop_token_abs.numel() == 0:
            return
        aux_idx = layer_idx
        if aux_idx >= self.aux_cache.n_layers:
            return
        # Only save tokens not already in the NEXT layer's KV or in this aux slot
        next_layer = layer_idx + 1
        if next_layer < self.kv_cache.n_layers:
            in_kv = self.kv_cache.cache_status_bit_array[next_layer]
        else:
            in_kv = torch.zeros_like(self.selected_tokens_bit_array)
        in_aux = self.aux_cache.cache_status_bit_array[aux_idx]
        pruned_tokens_bit_array = torch.zeros_like(self.selected_tokens_bit_array)
        pruned_tokens_bit_array[drop_token_abs] = True
        to_add_to_aux_bit_array = torch.logical_and(
            pruned_tokens_bit_array,
            torch.logical_not(torch.logical_or(in_kv, in_aux)),
        )
        to_add_to_aux_idxs = torch.nonzero(to_add_to_aux_bit_array).view(-1)
        if to_add_to_aux_idxs.numel() == 0:
            return
        aux_src = torch.index_select(
            self.hidden_states, 1, self.tkns_idxs_to_hidden_states_idxs[to_add_to_aux_idxs]
        )
        if aux_src.dtype != self.aux_cache.cache[aux_idx].dtype:
            aux_src = aux_src.to(self.aux_cache.cache[aux_idx].dtype)
        self.aux_cache.cache_status_bit_array[aux_idx].logical_or_(to_add_to_aux_bit_array)
        self.aux_cache.cache[aux_idx].index_copy_(1, to_add_to_aux_idxs, aux_src)

    def apply_pre_prune_from_prev_importance(self, layer_idx: int, pruning_rate: float) -> None:
        """
        根据上一层 scatter 的 token_importance_prev，在进入本层 Q/K/V 前裁剪 hidden_states。
        layer_idx==0 或 pruning_rate<=0 时不做；序列长度<=1 时不做。
        """
        if layer_idx == 0 or pruning_rate <= 0:
            return
        n = self.hidden_states_idxs.numel()
        if n <= 1:
            return
        k_drop = int(pruning_rate * n)
        k_drop = min(k_drop, n - 1)  # 至少保留 1 个 token（末 token 另用 inf 保护）
        if k_drop <= 0:
            return
        scores = self.token_importance_prev[self.hidden_states_idxs.long()].clone()
        scores = torch.nan_to_num(scores, nan=float("inf"))
        last_t = self.sequence_length - 1
        hi = self.tkns_idxs_to_hidden_states_idxs[last_t]
        scores[hi] = float("inf")
        keep_k = n - k_drop
        _, keep_rel = torch.topk(scores, keep_k, largest=True)
        keep_rel = torch.sort(keep_rel)[0]
        keep_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        keep_mask[keep_rel] = True
        drop_mask = ~keep_mask
        drop_abs = self.hidden_states_idxs[drop_mask]
        self.save_pre_prune_drops_to_aux(drop_abs, layer_idx)
        self.hidden_states = self.hidden_states[:, keep_rel, :]
        self.hidden_states_idxs = self.hidden_states_idxs[keep_rel]
        self.selected_tokens_bit_array[drop_abs] = False
        self._update_keys_idxs_to_tokens_idxs = True
        self._update_tkns_idxs_to_hidden_states_idxs = True

    @property
    def keys_idxs_to_tokens_idxs(self):
        """A mapping from the key's indexes to the token's indexes"""
        if self._update_keys_idxs_to_tokens_idxs:
            self._keys_idxs_to_tokens_idxs = torch.cat([self.in_kv_cache_idxs, self.hidden_states_idxs], dim=0)
            self._update_keys_idxs_to_tokens_idxs = False
        return self._keys_idxs_to_tokens_idxs

    @property
    def tkns_idxs_to_hidden_states_idxs(self):
        """A mapping from the token's indexes (positions in the sequence) to the hidden state's indexes"""
        if self._update_tkns_idxs_to_hidden_states_idxs:
            self._tkns_idxs_to_hidden_states_idxs = torch.empty(self.sequence_length, device=self.device, dtype=torch.long)
            self._tkns_idxs_to_hidden_states_idxs[self.hidden_states_idxs] = \
                torch.arange(self.hidden_states_idxs.shape[0], device=self.device, dtype=torch.long)
            self._update_tkns_idxs_to_hidden_states_idxs = False
        return self._tkns_idxs_to_hidden_states_idxs
    
    @property
    def hidden_states_bit_array(self):
        bit_array = torch.zeros_like(self.selected_tokens_bit_array)
        bit_array[self.hidden_states_idxs] = True
        return bit_array

    def get_kv_cache(self, layer_idx: int) -> HFCache:
        is_decode = self.hidden_states.shape[1] == 1
        
        if is_decode:
            # decode阶段：直接取该层所有已有KV，不用selected过滤
            in_kv_cache_idxs = torch.nonzero(
                self.kv_cache.cache_status_bit_array[layer_idx]
            ).view(-1)
            in_kv_cache_bit_array = self.kv_cache.cache_status_bit_array[layer_idx].clone()
        else:
            # prefill阶段：原来的逻辑
            in_kv_cache_bit_array = torch.logical_and(
                self.kv_cache.cache_status_bit_array[layer_idx],
                self.selected_tokens_bit_array
            )
            in_kv_cache_idxs = torch.nonzero(in_kv_cache_bit_array).view(-1)

        self.in_kv_cache_idxs = in_kv_cache_idxs
        self.in_kv_cache_bit_array = in_kv_cache_bit_array
        self._update_keys_idxs_to_tokens_idxs = True

        # NOTE:
        # 这里传给 HFCache 的 total_size 不能用“当前 key 的数量”。
        # 因为 valid_idxs / cache_position 是 token 的“绝对索引”(0..sequence_length-1)，剪枝后这些索引会变稀疏，
        # 此时 max(valid_idxs) 可能远大于 key 的数量，从而触发错误的 out-of-range 判断，甚至影响 HF attention 内部的长度检查。
        # 因此 total_size 在本项目中代表“逻辑上允许的最大 token 索引范围”(max index + 1)，一般等于当前序列长度。
        total_size = int(self.sequence_length)

        # 使用 KVCache 的最大长度作为三维长度，保证与全局缓冲区形状一致；
        # total_size 只作为“有效长度”元数据交给 HFCache 进行索引检查。
        local_kv_cache = HFCache(
            shape=(
                self.kv_cache.size[0], 
                self.kv_cache.size[1], 
                self.kv_cache.size[2],
                self.kv_cache.size[3]
            ),
            # ✅ 传入已有的 pre-allocated buffer 的 view，避免额外分配
            preallocated_key=self.kv_cache.key_cache[layer_idx],
            preallocated_value=self.kv_cache.value_cache[layer_idx],
            in_kv_cache_idxs=in_kv_cache_idxs,
            total_size=total_size,
            device=self.device,
            dtype=self.kv_cache.key_cache[layer_idx].dtype
        )
        return local_kv_cache

    def get_aux_cache(self, layer_idx: int):
        """Updates the hidden states with the tokens from the Aux Cache.

        Works in both prefill and decode phases:
        - Prefill: restores tokens that are in aux cache, selected, but not in KV cache.
        - Decode: same logic, but selected_tokens_bit_array covers all historical tokens
          so we restore any pruned token that is not yet in this layer's KV cache.
        """
        is_decode = self.hidden_states.shape[1] == 1

        if is_decode:
            # Decode phase: consider all tokens currently tracked (selected_tokens_bit_array)
            # that are in aux cache but NOT in this layer's KV cache.
            in_aux_cache_bit_array = torch.logical_and(
                self.aux_cache.cache_status_bit_array[layer_idx - 1],
                torch.logical_not(self.in_kv_cache_bit_array)
            )
        else:
            in_aux_cache_bit_array = torch.logical_and(
                torch.logical_and(self.aux_cache.cache_status_bit_array[layer_idx-1], self.selected_tokens_bit_array),
                # Removing those tokens that are in KV Cache
                torch.logical_not(self.in_kv_cache_bit_array)
            )
        in_aux_cache_idxs = torch.nonzero(in_aux_cache_bit_array).view(-1)
        
        self.hidden_states = torch.cat([
            self.hidden_states, 
            torch.index_select(self.aux_cache.cache[layer_idx-1], 1, in_aux_cache_idxs)
        ], dim=1)

        self.hidden_states_idxs = torch.cat([self.hidden_states_idxs, in_aux_cache_idxs], dim=0)
        # ✅ 修复：sequence_length 应该等于当前选中的最大 token 索引 + 1
        if in_aux_cache_idxs.shape[0] > 0:
            self.sequence_length = max(
                self.sequence_length,
                int(self.hidden_states_idxs.max().item()) + 1
            )
        self._update_keys_idxs_to_tokens_idxs = True
        self._update_tkns_idxs_to_hidden_states_idxs = True


    @property
    def hidden_states_positions(self):
        return self.tokens_positions_idxs[:, self.hidden_states_idxs]

    def update_kv_cache(self, local_kv_cache: HFCache, layer_idx: int, kv_cache_offset: int = None):
        # HFCache.update() 已经把 K/V 直接写进了全局 preallocated buffer（同一块显存），
        # 这里只需要更新 status bit，不再做重复的 index_copy_。
        in_hidden_states_bit_array = torch.logical_and(
            self.hidden_states_bit_array,
            torch.logical_not(self.kv_cache.cache_status_bit_array[layer_idx])
        )
        self.kv_cache.cache_status_bit_array[layer_idx].logical_or_(in_hidden_states_bit_array)
            
            
    '''只存那些：被剪掉 & 下一层KV Cache里没有 & Aux Cache里也没有的token'''
    def update_aux_cache(self, to_prune_idxs: torch.LongTensor, layer_idx: int):
        """Updates the Aux Cache with the hidden states of the pruned tokens"""
        in_next_layer_kv_bit_array = self.kv_cache.cache_status_bit_array[layer_idx+1]
        
        pruned_tokens_bit_array = torch.zeros_like(self.selected_tokens_bit_array)
        pruned_tokens_bit_array[to_prune_idxs] = True
        
        in_aux_cache_bit_array = self.aux_cache.cache_status_bit_array[layer_idx]

        to_add_to_aux_bit_array = torch.logical_and(
            pruned_tokens_bit_array, 
            # Removing those tokens that are in the next layer's KV Cache and those that are already in the Aux Cache
            torch.logical_not(torch.logical_or(in_next_layer_kv_bit_array, in_aux_cache_bit_array))
        ) 
        self.aux_cache.cache_status_bit_array[layer_idx].logical_or_(to_add_to_aux_bit_array)

        to_add_to_aux_idxs = torch.nonzero(to_add_to_aux_bit_array).view(-1)

        # Hidden states are stored in random order, so the tkns_idxs_to_hidden_states_idxs mapping is needed. 
        aux_src = torch.index_select(
            self.hidden_states, 1, self.tkns_idxs_to_hidden_states_idxs[to_add_to_aux_idxs]
        )
        if aux_src.dtype != self.aux_cache.cache[layer_idx].dtype:
            aux_src = aux_src.to(self.aux_cache.cache[layer_idx].dtype)
        self.aux_cache.cache[layer_idx].index_copy_(
            1,
            to_add_to_aux_idxs,
            aux_src
        )

    def prune(self, to_prune_idxs: torch.LongTensor):
        """Prunes the tokens from the hidden states"""
        self.selected_tokens_bit_array[to_prune_idxs] = False

        hidden_states_to_keep_bit_array = self.hidden_states_bit_array
        hidden_states_to_keep_bit_array[to_prune_idxs] = False
        hidden_states_to_keep_idxs = torch.nonzero(hidden_states_to_keep_bit_array).view(-1)

        # Hidden states are stored in random order, so the tkns_idxs_to_hidden_states_idxs mapping is needed.
        self.hidden_states = torch.index_select(self.hidden_states, 1, self.tkns_idxs_to_hidden_states_idxs[hidden_states_to_keep_idxs])

        self.hidden_states_idxs = hidden_states_to_keep_idxs

        self._update_keys_idxs_to_tokens_idxs = True
        self._update_tkns_idxs_to_hidden_states_idxs = True