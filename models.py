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

        dtype, device = inputs_embeds.dtype, inputs_embeds.device 
        batch_size = inputs_embeds.shape[0]

        # The cache_position tensor stores positions of hidden states in the sequence,
        # so the sequence length is the position of the last hidden state + 1  
        sequence_length = cache_position[-1].item() + 1

        # mask_kwargs = dict(
        #     config=self.config,
        #     inputs_embeds=inputs_embeds,
        #     attention_mask=attention_mask,
        #     cache_position=torch.arange(sequence_length, device=device),
        #     past_key_values=None,
        # )
        # causal_mask = {
        #     "full_attention": create_causal_mask(**mask_kwargs),
        # }
        # if self.config.use_sliding_window:
        #     causal_mask["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        
        # 改成直接生成一个tensor，不用dict
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=torch.arange(sequence_length, device=device),
            past_key_values=None,
        )

        context = Context(
            inputs_embeds,
            kv_cache,
            aux_cache,
            position_ids,
            cache_position,
            sequence_length,
        )

        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                context,
                causal_mask,
                self.rotary_emb,
                output_attentions,
            )
            context = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        context.hidden_states = self.norm(context.hidden_states)
        return context.hidden_states, all_self_attns
    
    def from_qwen3_state_dict(qwen3_state_dict, config, pruning_rates=None):
        if isinstance(config, Qwen3Config):
            config = LazyQwen3Config.from_qwen3_config(pruning_rates, config)
        new_state_dict = OrderedDict((modify_key(k), v) for k, v in qwen3_state_dict.items())
        model = LazyQwen3ForCausalLM(config)
        model.load_state_dict(new_state_dict)
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
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        # print(f"[DEBUG] logits shape: {logits.shape}")

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
        output_sequence = input_ids

        batch_size = input_ids.shape[0]
        embed_size_per_head = self.config.head_dim
        
        if logits_processor is None:
            logits_processor = LogitsProcessorList()

        kv_cache = KVCache(
            self.config.num_hidden_layers,
            batch_size,
            self.config.num_key_value_heads,
            max_length,
            embed_size_per_head,
            input_ids.device,
        )

        aux_cache = AuxCache(
            self.config.num_hidden_layers,
            batch_size,
            max_length,
            self.config.hidden_size,
            input_ids.device,
        )

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
        while cache_position[-1].item() < max_length and not torch.all(input_ids[:, -1] == eos_token_id):
            torch.cuda.synchronize()
            t_start = time.time()
            outputs = self(**model_inputs)
            # ← 加计时结束
            torch.cuda.synchronize()
            t_end = time.time()
            
            # ← 记录时间
            if is_prefill:
                top5 = torch.topk(outputs[0][:, -1, :], 5)
                print(f"[PREFILL TOP5] values: {top5.values}")
                print(f"[PREFILL TOP5] indices: {top5.indices}")
                prefill_time = t_end - t_start
                is_prefill = False
                print(f"[TIMER] Prefill时间(TTFT): {prefill_time*1000:.1f}ms, 输入tokens: {cache_position[-1].item()+1}")
            else:
                decode_times.append(t_end - t_start)
            
            next_token_logits = outputs[0][:, -1, :].clone()
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
            print(f"[TIMER] 总时间(TTFT+Decode): {(prefill_time*1000+total_decode):.1f}ms")
    
        return output_sequence
    
    def from_qwen3_state_dict(qwen3_state_dict, config, pruning_rates=None):
        if isinstance(config, Qwen3Config):
            config = LazyQwen3Config.from_qwen3_config(pruning_rates, config)
        new_state_dict = OrderedDict((modify_key(k), v) for k, v in qwen3_state_dict.items())
        model = LazyQwen3ForCausalLM(config)  # ← 注意这里
        model.load_state_dict(new_state_dict)
        return model