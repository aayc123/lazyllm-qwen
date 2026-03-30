from typing import Optional, Union
from transformers import PreTrainedModel, LogitsProcessorList
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm, Qwen3RotaryEmbedding
)
from transformers import Qwen3ForCausalLM, Qwen3Config, AutoTokenizer
from config import LazyQwen3Config 
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from config import LazyQwen3Config
import torch.nn as nn
import torch
from decoder_layer import DecoderLayer
from caches import KVCache, AuxCache
from context import Context
from collections import OrderedDict
import time
import os

def modify_key(key):
    if "model.layers" in key:
        temp = key.split(".")
        temp.insert(3, "decoder")
        return ".".join(temp)
    else:
        return key

class LazyQwen3Model(PreTrainedModel):
    """
    A custom decoder-based model that builds upon the Qwen3Model and implements dynamic token pruning.

    This is an implementation of "LazyLLM: DYNAMIC TOKEN PRUNING FOR EFFICIENT LONG CONTEXT LLM INFERENCE"
    with Qwen3 as the base model.
    """
    def __init__(self, config: LazyQwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        
    
    def forward(
            self,
            kv_cache: KVCache,
            aux_cache: AuxCache,
            cache_position: torch.LongTensor,
            input_ids: torch.LongTensor,
            attention_mask: torch.Tensor,
            position_ids: torch.LongTensor,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
        ):
        """
        Executes the forward pass for the model, updating the hidden states and caches,

        Args:
            kv_cache (KVCache): The key-value cache.
            aux_cache (AuxCache): The aux cache.
            cache_position (torch.LongTensor): The position of the hidden states in the sequence. Same as the `cache_position`
                in the original code.
            input_ids (torch.LongTensor): The input token IDs of shape (batch_size, sequence_length).
            attention_mask (torch.Tensor): The 2D attention mask of shape (batch_size, sequence_length).
            position_ids (torch.LongTensor): The position IDs for the whole sequence. Note that in the original code, this
                argument was only storing the positions of the current hidden states, not the whole sequence.
            inputs_embeds (torch.FloatTensor): Optional input embeddings. Can be used instead of `input_ids`.
            output_attentions (bool): Whether to return attention weights.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if inputs_embeds.shape[1] > 1:
            print(f"[position_ids] shape={position_ids.shape}, first5={position_ids[0,:5].tolist()}, last5={position_ids[0,-5:].tolist()}")
        dtype, device = inputs_embeds.dtype, inputs_embeds.device 
        batch_size = inputs_embeds.shape[0]

        # The cache_position tensor stores positions of hidden states in the sequence,
        # so the sequence length is the position of the last hidden state + 1  
        sequence_length = cache_position[-1].item() + 1
        mask_full_length = attention_mask.shape[1]
        # 改成直接生成一个tensor，不用dict
        is_decode = inputs_embeds.shape[1] == 1

        if is_decode:
            # decode阶段：当前单token可以attend到所有past tokens
            # shape: (batch, 1, 1, full_len)
            causal_mask = torch.zeros(
                (batch_size, 1, 1, mask_full_length),
                dtype=torch.float32,
                device=device,
            )
            # padding位置mask掉
            if attention_mask is not None:
                padding_mask = (attention_mask == 0).float() * -1e4  # (B, full_len)
                causal_mask[:, :, 0, :] = padding_mask
            use_sw = False
            sliding_window_mask = None
        else:
            causal_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=torch.arange(mask_full_length, device=device),
                past_key_values=None,
            )
            if causal_mask.dtype != torch.float32:
                causal_mask = causal_mask.to(torch.float32)
            causal_mask = causal_mask.clamp(min=-1e4)
            use_sw = (
                hasattr(self.config, 'use_sliding_window') and
                self.config.use_sliding_window and
                self.config.sliding_window is not None
            )
            sliding_window_mask = None
            if use_sw:
                sliding_window_mask = create_sliding_window_causal_mask(
                    config=self.config,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    cache_position=torch.arange(mask_full_length, device=device),
                    past_key_values=None,
                    sliding_window=self.config.sliding_window,
                )
                if sliding_window_mask.dtype != torch.float32:
                    sliding_window_mask = sliding_window_mask.to(torch.float32)
                sliding_window_mask = sliding_window_mask.clamp(min=-1e4)
        if inputs_embeds.shape[1] > 1:
            print(f"[causal_mask] shape={causal_mask.shape}, ...")
        
        context = Context(
            inputs_embeds, kv_cache, aux_cache,
            position_ids, cache_position, sequence_length,
        )
        
        all_self_attns = () if output_attentions else None
        
        # ← 关键：合并成一个循环，按层选择对应mask
        for layer_idx, decoder_layer in enumerate(self.layers):
            if use_sw and layer_idx >= self.config.max_window_layers:
                layer_causal_mask = sliding_window_mask
            else:
                layer_causal_mask = causal_mask
            
            layer_outputs = decoder_layer(
                context,
                layer_causal_mask,   # ← 传对应层的mask
                self.rotary_emb,
                output_attentions,
            )
            context = layer_outputs[0]
            
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        
        context.hidden_states = self.norm(context.hidden_states)
        self.last_hidden_states_idxs = context.hidden_states_idxs
        return context.hidden_states, all_self_attns
        
    def from_qwen3_state_dict(qwen3_state_dict, config, pruning_rates=None):
        if isinstance(config, Qwen3Config):
            config = LazyQwen3Config.from_qwen3_config(pruning_rates, config)
        
        new_state_dict = OrderedDict(
            (modify_key(k), v) for k, v in qwen3_state_dict.items()
        )
        
        # to_empty在CPU上分配空内存（不做随机初始化，比普通init省一半时间）
        with torch.device("meta"):
            model = LazyQwen3ForCausalLM(config)
        
        model = model.to_empty(device="cpu")
        model.load_state_dict(new_state_dict, assign=True)
        del new_state_dict
        import gc; gc.collect()

        # meta -> to_empty 会让“未出现在 state_dict 里的 buffer”变成未初始化内存。
        # Qwen3RotaryEmbedding 的 inv_freq/cos/sin cache 等属于这类 buffer，
        # 若不重建，会导致 rotary_emb 输出 cos/sin=NaN，进而整网 NaN。
        model.model.rotary_emb = Qwen3RotaryEmbedding(config=model.config)
        
        return model
        
class LazyQwen3ForCausalLM(PreTrainedModel):
    """
    A custom decoder-based model that builds upon the Qwen3Model and implements dynamic token pruning.

    This is an implementation of "LazyLLM: DYNAMIC TOKEN PRUNING FOR EFFICIENT LONG CONTEXT LLM INFERENCE"
    with Qwen3 as the base model. It is specifically designed for causal language modeling tasks and it
    implements a custom generate method.
    """
    def __init__(self, config):
        super().__init__(config)
        self.model = LazyQwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
            self,
            kv_cache: KVCache,
            aux_cache: AuxCache,
            cache_position: torch.LongTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
        ):
        """
        Executes the forward pass for the model, updating the hidden states and caches,

        Args:
            kv_cache (KVCache): The key-value cache.
            aux_cache (AuxCache): The aux cache.
            cache_position (torch.LongTensor): The position of the hidden states in the sequence. Same as the `cache_position`
                in the original code.
            input_ids (torch.LongTensor): The input token IDs of shape (batch_size, sequence_length).
            attention_mask (torch.Tensor): The 2D attention mask of shape (batch_size, sequence_length).
            position_ids (torch.LongTensor): The position IDs for the whole sequence. Note that in the original code, this
                argument was only storing the positions of the current hidden states, not the whole sequence.
            inputs_embeds (torch.FloatTensor): Optional input embeddings. Can be used instead of `input_ids`.
            output_attentions (bool): Whether to return attention weights.
        """
        outputs = self.model(
            kv_cache=kv_cache,
            aux_cache=aux_cache,
            cache_position=cache_position,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
        )

        hidden_states = outputs[0] 
        if hidden_states.shape[1] > 1:
            hidden_states_idxs = self.model.last_hidden_states_idxs  # 从model拿到idxs
            last_pos_in_hidden = (hidden_states_idxs == hidden_states_idxs.max()).nonzero(as_tuple=True)[0][0]
            last_token_hs = hidden_states[:, last_pos_in_hidden:last_pos_in_hidden+1, :]
            # print(f"[lm_head input] shape={hidden_states.shape}, last token stats: mean={last_token_hs[0,0,:].mean().item():.4f}, std={last_token_hs[0,0,:].std().item():.4f}, max={last_token_hs[0,0,:].max().item():.4f}")
            # print(f"[lm_head debug] hidden_states_idxs={hidden_states_idxs.tolist()}, last_pos_in_hidden={last_pos_in_hidden.item()}, max_idx={hidden_states_idxs.max().item()}")
            logits = self.lm_head(last_token_hs)  # (1, 1, vocab_size)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        return logits, outputs[1] if output_attentions else None
    
    def generate(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.Tensor,
            max_length: int,
            eos_token_id: int,
            pad_token_id: int,
            output_attentions: Optional[bool] = False, 
            logits_processor: Optional[LogitsProcessorList] = None,
            do_sample: Optional[bool] = False,
            preallocated_kv_cache: Optional[object] = None,
            preallocated_aux_cache: Optional[object] = None,
            return_scores: Optional[bool] = False,
        ) -> torch.LongTensor:
        """
        Generates a sequence of tokens from a given prompt. It can be used for both greedy and sampling-based decoding.

        Args:
            input_ids (torch.LongTensor): The input token IDs of shape (batch_size, sequence_length).
            attention_mask (torch.Tensor): The 2D attention mask of shape (batch_size, sequence_length).
            max_length (int): The maximum length of the generated sequence. This must be provided since dynamic token pruning
                relies on the maximum length of the sequence for allocating memory for caches.
            eos_token_id (int): The end of a sequence token ID.
            pad_token_id (int): The padding token ID.
            output_attentions (Optional[bool]): Whether to return attention weights.
            logits_processor (Optional[LogitsProcessorList]): A list of logits processors to apply to the logits.
            do_sample (Optional[bool]): Whether to use sampling-based decoding or not.
        """
        # 每个 decode step 开始时
        
        output_sequence = input_ids
        scores = [] if return_scores else None

        batch_size = input_ids.shape[0]
        embed_size_per_head = self.config.head_dim
        
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        
        input_len = input_ids.shape[1]
        max_new_tokens = max_length - input_len
        # 限制最多生成200个新token（按你的dataset2maxlen来）
        max_new_tokens_capped = min(int(max_new_tokens), 200)
        effective_max_length = int(input_len + max_new_tokens_capped)
        # 以 effective_max_length 作为本次生成的真实上限，确保缓存分配/while 条件一致
        max_length = effective_max_length
        force_attn_fp32 = os.getenv("LAZYLLAMA_ATTN_FP32", "0") == "1"
        if preallocated_kv_cache is not None and not force_attn_fp32:
            kv_cache = preallocated_kv_cache
            kv_cache.reset()
            if int(kv_cache.size[2]) < int(effective_max_length):
                raise ValueError(
                    f"preallocated_kv_cache length={int(kv_cache.size[2])} is smaller than required max_length={int(effective_max_length)}. "
                    "Increase MAX_SEQ_LEN (prompt+generation) or reduce max_length/max_gen."
                )
        else:
            kv_cache = KVCache(
                self.config.num_hidden_layers,
                batch_size,
                self.config.num_key_value_heads,
                effective_max_length,
                embed_size_per_head,
                input_ids.device,
                dtype=torch.float32 if force_attn_fp32 else torch.float32
            )
        if preallocated_aux_cache is not None and not force_attn_fp32:
            aux_cache = preallocated_aux_cache
            aux_cache.reset()
            if int(aux_cache.size[1]) < int(effective_max_length):
                raise ValueError(
                    f"preallocated_aux_cache length={int(aux_cache.size[1])} is smaller than required max_length={int(effective_max_length)}. "
                    "Increase MAX_SEQ_LEN (prompt+generation) or reduce max_length/max_gen."
                )
        else:
            aux_cache = AuxCache(
                self.config.num_hidden_layers,
                batch_size,
                effective_max_length,
                self.config.hidden_size,
                input_ids.device,
                dtype=torch.float32 if force_attn_fp32 else torch.float32
            )
        # print(f"KVCache dtype: {kv_cache.key_cache[0].dtype}")
        # print(f"AuxCache dtype: {aux_cache.cache[0].dtype}")
        cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)

        # Creating position_ids on the fly. The default value (for padding tokens) is 1
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "kv_cache": kv_cache,
            "aux_cache": aux_cache,
            "cache_position": cache_position,
            "position_ids": position_ids,
            "output_attentions": output_attentions, 
        }
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        is_prefill = True  # ← 加这行
        prefill_time = 0   # ← 加这行
        decode_times = []  # ← 加这行
        # is_first_step = True 
        
        while cache_position[-1].item() < max_length and not torch.all(input_ids[:, -1] == eos_token_id):
            torch.cuda.synchronize()
            t_start = time.time()
            with torch.no_grad():
                outputs = self(**model_inputs)
                # ← 加计时结束
                torch.cuda.synchronize()
                t_end = time.time()
                print(f"[step {'prefill' if is_prefill else len(decode_times)}] allocated: {torch.cuda.memory_allocated()/1024**3:.1f}GB, reserved: {torch.cuda.memory_reserved()/1024**3:.1f}GB")

                # ← 记录时间
                if is_prefill:
                    top5 = torch.topk(outputs[0][:, -1, :], 5)
                    # print(f"[PREFILL TOP5] values: {top5.values}")
                    # print(f"[PREFILL TOP5] indices: {top5.indices}")
                    prefill_time = t_end - t_start
                    is_prefill = False
                    print(f"[TIMER] Prefill时间(TTFT): {prefill_time*1000:.1f}ms, 输入tokens: {cache_position[-1].item()+1}")
                else:
                    decode_times.append(t_end - t_start)
                # if is_first_step:
                #     kv = model_inputs['kv_cache']
                #     aux = model_inputs['aux_cache']
                #     is_first_step = False
                next_token_logits = outputs[0][:, -1, :].clone()
                if return_scores:
                    # 将当前步 logits 存起来，用于与 Baseline 精确对齐分析
                    scores.append(next_token_logits.detach().cpu())
                # 找到这段诊断代码，替换为：
                if is_prefill or len(decode_times) < 2:
                    # print(f"[logits] max={next_token_logits.max().item():.2f}, min={next_token_logits.min().item():.2f}, has_nan={next_token_logits.isnan().any().item()}, has_inf={next_token_logits.isinf().any().item()}")
                    top3 = torch.topk(next_token_logits[0], 3)
                    # print(f"[logits] top3 ids={top3.indices.tolist()}, vals={[f'{v:.2f}' for v in top3.values.tolist()]}")
                next_token_scores = logits_processor(input_ids, next_token_logits)
                
                if do_sample:
                    probs = nn.functional.softmax(next_token_scores, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_scores, dim=-1)

                # Finished sentences should have their next token be a padding token
                next_tokens = next_tokens * unfinished_sequences + (1 - unfinished_sequences) * pad_token_id

                unfinished_sequences.mul_(next_tokens != eos_token_id)

                # Updating model inputs for the next generation step
                input_ids = next_tokens.view(-1, 1) 
                output_sequence = torch.cat([output_sequence, input_ids], dim=-1)             
                cache_position = torch.tensor([cache_position[-1] + 1], device=cache_position.device)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((batch_size, 1), device=attention_mask.device)], 
                    dim=-1
                )
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)

                model_inputs.update({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "cache_position": cache_position,
                })
        if decode_times:
            avg_decode = sum(decode_times) / len(decode_times) * 1000
            total_decode = sum(decode_times) * 1000
            print(f"[TIMER] Decode平均每token: {avg_decode:.1f}ms")
            print(f"[TIMER] Decode总时间: {total_decode:.1f}ms, 共{len(decode_times)}个tokens")
            print(f"[TIMER] Prefill时间(TTFT): {prefill_time*1000:.1f}ms")
            print(f"[TIMER] 总时间(TTFT+Decode): {(prefill_time*1000+total_decode):.1f}ms")
            
        model_inputs.clear()

        # 只有自己创建的才释放，预分配的不释放
        if preallocated_kv_cache is None:
            kv_cache.key_cache.clear()
            kv_cache.value_cache.clear()
            kv_cache.cache_status_bit_array = None
            aux_cache.cache.clear()
            aux_cache.cache_status_bit_array = None
            del kv_cache, aux_cache

        import gc; gc.collect()
        torch.cuda.empty_cache()
        if return_scores:
            return output_sequence, scores
        return output_sequence
    
    def from_qwen3_state_dict(qwen3_state_dict, config, pruning_rates=None):
        if isinstance(config, Qwen3Config):
            config = LazyQwen3Config.from_qwen3_config(pruning_rates, config)
        
        new_state_dict = OrderedDict(
            (modify_key(k), v) for k, v in qwen3_state_dict.items()
        )
        
        # to_empty在CPU上分配空内存（不做随机初始化，比普通init省一半时间）
        with torch.device("meta"):
            model = LazyQwen3ForCausalLM(config)
        
        model = model.to_empty(device="cpu")
        model.load_state_dict(new_state_dict, assign=True)
        del new_state_dict
        import gc; gc.collect()

        # meta -> to_empty 会让未出现在 state_dict 里的 buffer 处于未初始化状态。
        # rotary_emb 的 inv_freq/cos/sin cache 属于 buffer，若不重建会导致 RoPE 输出 NaN。
        model.model.rotary_emb = Qwen3RotaryEmbedding(config=model.config)
        
        return model
        