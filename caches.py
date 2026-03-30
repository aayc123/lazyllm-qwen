import torch
from transformers import StaticCache
from typing import Tuple, Optional

class KVCache:
    """A static storage for the KV caches of all layers, preserving the positions of the caches in the sequence."""
    def __init__(
            self, 
            n_layers: int, 
            batch_size: int, 
            num_heads: int, 
            sequence_length: int, 
            embed_size_per_head: int, 
            device: torch.device,
            dtype=torch.float32
        ):
        """
        Initializes a KVCache to store key and value caches for all layers.

        Args:
            n_layers (int): Number of layers in the transformer model.
            batch_size (int): Number of batches.
            num_heads (int): Number of attention heads.
            sequence_length (int): The (maximal) length of the input sequence.
            embed_size_per_head (int): Embedding size per attention head.
            device (torch.device): Device to store the tensors (e.g., 'cpu' or 'cuda').
            dtype (torch.dtype): Data type for the tensors.
        """
        self.size = (batch_size, num_heads, sequence_length, embed_size_per_head)
        self.n_layers = n_layers
        self.key_cache = [torch.zeros(self.size, device=device, dtype=dtype) for _ in range(n_layers)]
        self.value_cache = [torch.zeros(self.size, device=device, dtype=dtype) for _ in range(n_layers)]
        self.cache_status_bit_array = torch.zeros((n_layers, sequence_length), dtype=torch.bool, device=device)
    
    def reset(self):
            """重置cache状态，复用已分配的显存"""
            self.cache_status_bit_array.zero_()
            
class AuxCache:
    """The Aux Cache stores hidden states of pruned tokens that are not present in the subsequent layers' KV caches."""
    def __init__(
            self, 
            n_layers: int, 
            batch_size: int, 
            sequence_length: int, 
            hidden_size: int, 
            device: torch.device,
            dtype=torch.float32
        ):
        """
        Initializes an AuxCache to store hidden states of pruned tokens.
        
        Args:
            n_layers (int): Number of layers in the transformer model.
            batch_size (int): Number of batches.
            sequence_length (int): The (maximal) length of the input sequence.
            hidden_size (int): Size of the hidden state vectors.
            device (torch.device): Device to store the tensors (e.g., 'cpu' or 'cuda').
            dtype (torch.dtype): Data type for the tensors.
        """
        self.size = (batch_size, sequence_length, hidden_size)
        self.n_layers = n_layers-1
        self.cache = [torch.zeros(self.size, device=device,dtype=dtype) for _ in range(n_layers-1)]
        self.cache_status_bit_array = torch.zeros((n_layers-1, sequence_length), dtype=torch.bool, device=device)

    def reset(self):
            """重置cache状态，复用已分配的显存"""
            self.cache_status_bit_array.zero_()
            # cache tensor 不需要清零，status_bit_array 为 False 时会被覆盖写入
            
class HFCache:
    """
    替代原来继承StaticCache的版本，只实现Qwen3Attention.forward需要的update()方法
    """
    def __init__(
            self, 
            shape: Tuple[int], 
            device: torch.device, 
            dtype: torch.float32,
            cache: Optional[Tuple[torch.FloatTensor]] = None,
            preallocated_key=None,   # ← 新增
            preallocated_value=None, # ← 新增
            in_kv_cache_idxs=None,   # ← 新增
            total_size=None,         # ← 新增
        ):
        # shape: (batch_size, num_heads, max_cache_len, embed_size_per_head)
        # 这里的 max_cache_len 应该等于全局 KVCache 预分配的最大长度
        self.max_cache_len = shape[2]
        
        if preallocated_key is not None:
            # ✅ 引用全局 KVCache 的 tensor 作为底层存储
            # 注意：它的第三维长度是全局 max_cache_len，而不是当前 step 的 total_size
            self._key_cache = preallocated_key   # shape: (B, H, max_cache_len, D)
            self._value_cache = preallocated_value
            # 当前已经在 KVCache 中的 token 数（由 Context 计算传入）
            self._current_len = in_kv_cache_idxs.shape[0]
            # 逻辑上的“最大允许 token 索引范围”(max index + 1)。
            # 不能用“有效 key 的数量”，因为剪枝会让 token 索引变稀疏（max_idx 可能远大于 key 数量）。
            self._total_size = int(total_size) if total_size is not None else None
            self._in_kv_cache_idxs = in_kv_cache_idxs
        elif cache is None:
            self._key_cache = torch.zeros(shape, device=device,dtype=dtype)
            self._value_cache = torch.zeros(shape, device=device,dtype=dtype)
            self._current_len = 0
        else:
            existing_len = cache[0].shape[2]
            actual_dtype = cache[0].dtype
            pad_len = shape[2] - existing_len
            pad_shape = (shape[0], shape[1], pad_len, shape[3])
            self._key_cache = torch.cat(
                [cache[0], torch.zeros(pad_shape, device=device, dtype=actual_dtype)], dim=2
            )
            self._value_cache = torch.cat(
                [cache[1], torch.zeros(pad_shape, device=device, dtype=actual_dtype)], dim=2
            )
            self._current_len = existing_len

        self.key_cache = [self._key_cache]
        self.value_cache = [self._value_cache]
        self.seen_tokens = self._current_len

    def update(self, key_states, value_states, layer_idx, cache_kwargs):
        # 强制转换，防御性处理
        key_states = key_states.to(self._key_cache.dtype)
        value_states = value_states.to(self._value_cache.dtype)
        
        cache_position = cache_kwargs["cache_position"]
        # 打印前几层的 KV 更新情况，确认是否所有层都在正确写入
        if layer_idx < 3 and cache_position.shape[0] > 1:
            print(f"[HFCache.update] layer={layer_idx}, cache_position={cache_position}, key_states.shape={key_states.shape}")
        # if key_states.shape[2] > 1:
        #     print(f"[HFCache.update PREFILL] layer={layer_idx}")
        #     print(f"  key_states.shape={key_states.shape}")
        #     print(f"  cache_position.shape={cache_position.shape}, first5={cache_position[:5].tolist()}, last5={cache_position[-5:].tolist()}")
        #     print(f"  _in_kv_cache_idxs.shape={self._in_kv_cache_idxs.shape}")
        self._key_cache[:, :, cache_position, :] = key_states
        self._value_cache[:, :, cache_position, :] = value_states

        self._current_len += key_states.shape[2]
        self.seen_tokens = self._current_len

        # 当前 KV 中有效的序列位置索引
        valid_idxs = torch.cat([self._in_kv_cache_idxs, cache_position])

        # 仅在前几层和 prefill 阶段做索引健全性检查，避免刷屏/开销过大
        if layer_idx < 3 and cache_position.shape[0] > 1:
            num_total = valid_idxs.numel()
            unique_idxs = torch.unique(valid_idxs)
            num_unique = unique_idxs.numel()
            num_dup = int(num_total - num_unique)
            min_idx = int(valid_idxs.min()) if num_total > 0 else -1
            max_idx = int(valid_idxs.max()) if num_total > 0 else -1
            physical_max = int(self.max_cache_len - 1)
            if hasattr(self, "_total_size") and self._total_size is not None:
                logical_max = int(self._total_size - 1)
                max_allowed = int(min(physical_max, logical_max))
            else:
                max_allowed = physical_max
            print(
                f"[HFCache.update idxcheck] layer={layer_idx}, total={num_total}, unique={num_unique}, "
                f"dup={num_dup}, min_idx={min_idx}, max_idx={max_idx}, max_allowed={max_allowed}"
            )
            if (valid_idxs < 0).any() or (valid_idxs > max_allowed).any():
                print(f"[HFCache.update WARNING] layer={layer_idx}, out-of-range index detected in valid_idxs")

        return self._key_cache[:, :, valid_idxs, :], self._value_cache[:, :, valid_idxs, :]
    
    def get_seq_length(self, layer_idx=0):
        return self._current_len

    def get_max_length(self):
        # HF attention 会用 get_max_length() 来判断 cache_position 是否越界；
        # 这里应该返回“允许的最大 token 索引范围”，受物理容量和逻辑序列长度共同约束。
        if hasattr(self, "_total_size") and self._total_size is not None:
            return int(min(self.max_cache_len, self._total_size))
        return int(self.max_cache_len)