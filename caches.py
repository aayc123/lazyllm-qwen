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
        # shape: (batch_size, num_heads, sequence_length, embed_size_per_head)
        self.max_cache_len = shape[2]
        
        if preallocated_key is not None:
            # ✅ 零拷贝：直接引用全局 KVCache 的 tensor
            # 注意：这里是 view，不是 copy
            self._key_cache = preallocated_key   # shape: (B, H, max_len, D)
            self._value_cache = preallocated_value
            self._current_len = in_kv_cache_idxs.shape[0]
            self._total_size = total_size
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
        if key_states.shape[2] > 1:
            print(f"[HFCache.update PREFILL] layer={layer_idx}")
            print(f"  key_states.shape={key_states.shape}")
            print(f"  cache_position.shape={cache_position.shape}, first5={cache_position[:5].tolist()}, last5={cache_position[-5:].tolist()}")
            print(f"  _in_kv_cache_idxs.shape={self._in_kv_cache_idxs.shape}")
        self._key_cache[:, :, cache_position, :] = key_states
        self._value_cache[:, :, cache_position, :] = value_states

        self._current_len += key_states.shape[2]
        self.seen_tokens = self._current_len
        valid_idxs = torch.cat([self._in_kv_cache_idxs, cache_position])
        return self._key_cache[:, :, valid_idxs, :], self._value_cache[:, :, valid_idxs, :]
    
    def get_seq_length(self, layer_idx=0):
        return self._current_len

    def get_max_length(self):
        return self._total_size if hasattr(self, '_total_size') else self.max_cache_len