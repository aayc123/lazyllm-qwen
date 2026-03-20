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
            device: torch.device
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
        """
        self.size = (batch_size, num_heads, sequence_length, embed_size_per_head)
        self.n_layers = n_layers
        self.key_cache = tuple(torch.zeros(self.size, device=device) for _ in range(n_layers))
        self.value_cache = tuple(torch.zeros(self.size, device=device) for _ in range(n_layers))
        self.cache_status_bit_array = torch.zeros((n_layers, sequence_length), dtype=torch.bool, device=device)

class AuxCache:
    """The Aux Cache stores hidden states of pruned tokens that are not present in the subsequent layers' KV caches."""
    def __init__(
            self, 
            n_layers: int, 
            batch_size: int, 
            sequence_length: int, 
            hidden_size: int, 
            device: torch.device
        ):
        """
        Initializes an AuxCache to store hidden states of pruned tokens.
        
        Args:
            n_layers (int): Number of layers in the transformer model.
            batch_size (int): Number of batches.
            sequence_length (int): The (maximal) length of the input sequence.
            hidden_size (int): Size of the hidden state vectors.
            device (torch.device): Device to store the tensors (e.g., 'cpu' or 'cuda').
        """
        self.size = (batch_size, sequence_length, hidden_size)
        self.n_layers = n_layers-1
        self.cache = tuple(torch.zeros(self.size, device=device) for _ in range(n_layers-1))
        self.cache_status_bit_array = torch.zeros((n_layers-1, sequence_length), dtype=torch.bool, device=device)

class HFCache:
    """
    替代原来继承StaticCache的版本，只实现Qwen3Attention.forward需要的update()方法
    """
    def __init__(
            self, 
            shape: Tuple[int], 
            device: torch.device, 
            cache: Optional[Tuple[torch.FloatTensor]] = None
        ):
        # shape: (batch_size, num_heads, sequence_length, embed_size_per_head)
        self.max_cache_len = shape[2]
        
        if cache is None:
            self._key_cache = torch.zeros(shape, device=device)
            self._value_cache = torch.zeros(shape, device=device)
            self._current_len = 0
        else:
            # cache[0]: (batch, heads, existing_len, head_dim)
            existing_len = cache[0].shape[2]
            pad_len = shape[2] - existing_len
            pad_shape = (shape[0], shape[1], pad_len, shape[3])
            self._key_cache = torch.cat([cache[0], torch.zeros(pad_shape, device=device)], dim=2)
            self._value_cache = torch.cat([cache[1], torch.zeros(pad_shape, device=device)], dim=2)
            self._current_len = existing_len

        # 兼容原来代码里访问 local_kv_cache.key_cache[0] 的地方
        self.key_cache = [self._key_cache]
        self.value_cache = [self._value_cache]

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,          # Qwen3Attention传进来的，这里忽略（每个HFCache只管一层）
            cache_kwargs: dict,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        把新的key/value写入cache，返回完整的key/value（历史+新的）
        cache_position 指示新token写入的位置
        """
        cache_position = cache_kwargs["cache_position"]  # shape: (new_token_len,)
        
        self._key_cache[:, :, cache_position, :] = key_states
        self._value_cache[:, :, cache_position, :] = value_states

        # 返回到目前为止所有有效的key/value
        new_len = cache_position.max().item() + 1
        return self._key_cache[:, :, :new_len, :], self._value_cache[:, :, :new_len, :]