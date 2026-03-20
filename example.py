import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from transformers import Qwen3ForCausalLM, AutoTokenizer
from models import *
import torch
import time
from caches import HFCache
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
local_path = "/data/zn/model/models/Qwen3-8B"

qwen3_model = Qwen3ForCausalLM.from_pretrained(
    local_path, 
    torch_dtype=torch.bfloat16,
    attn_implementation="eager"
).to(device)
tokenizer = AutoTokenizer.from_pretrained(local_path)

lazy_qwen3_model = LazyQwen3ForCausalLM.from_qwen3_state_dict(
    qwen3_model.state_dict(),
    qwen3_model.config,
    pruning_rates={i: 0.1 for i in range(qwen3_model.config.num_hidden_layers)},
).to(device)

prompt = 'Write a delicious recipe for french fries.\n\n'
input_sequence = tokenizer([prompt], return_tensors="pt")
input_ids = input_sequence["input_ids"].to(device)
attention_mask = input_sequence["attention_mask"].to(device)

# ---- 原始Qwen3 Prefill计时 ----
# 先warmup一次，避免第一次运行的cuda初始化影响计时
with torch.no_grad():
    _ = qwen3_model(input_ids, attention_mask=attention_mask)

torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    _ = qwen3_model(input_ids, attention_mask=attention_mask)
torch.cuda.synchronize()
t1 = time.time()
# print(f"[原始Qwen3] Prefill时间(TTFT): {(t1-t0)*1000:.1f}ms, 输入tokens: {input_ids.shape[1]}")

# ---- 原始Qwen3 生成计时 ----
torch.cuda.synchronize()
t2 = time.time()
output_ids = qwen3_model.generate(input_ids, max_new_tokens=200)
torch.cuda.synchronize()
t3 = time.time()
output_len = output_ids.shape[1] - input_ids.shape[1]
# print(f"[原始Qwen3] 生成总时间: {(t3-t2)*1000:.1f}ms, 输出tokens: {output_len}, 平均每token: {(t3-t2)*1000/output_len:.1f}ms")
# print(f"[原始Qwen3] 输出:\n{tokenizer.decode(output_ids[0], skip_special_tokens=True)}\n")

# ---- LazyQwen3 生成计时（prefill在generate内部打印）----
torch.cuda.synchronize()
t4 = time.time()
original_forward = lazy_qwen3_model.forward

hf_call_count = [0]
forward_call_count = [0]

# def update(self, key_states, value_states, layer_idx, cache_kwargs):
#     cache_position = cache_kwargs["cache_position"]
    
#     hf_call_count[0] += 1
#     if hf_call_count[0] <= 5 and layer_idx == 0:
#         print(f"[HFCACHE update #{hf_call_count[0]}] cache_position: {cache_position}")
#         print(f"[HFCACHE update #{hf_call_count[0]}] key_states shape: {key_states.shape}")
#         print(f"[HFCACHE update #{hf_call_count[0]}] _key_cache shape: {self._key_cache.shape}")
#         new_len = cache_position[-1].item() + 1
#         print(f"[HFCACHE update #{hf_call_count[0]}] 返回的key长度: {new_len}")
    
#     self._key_cache[:, :, cache_position, :] = key_states
#     self._value_cache[:, :, cache_position, :] = value_states
    
#     new_len = cache_position[-1].item() + 1
#     return self._key_cache[:, :, :new_len, :], self._value_cache[:, :, :new_len, :]

# HFCache.update = update

def debug_forward(*args, **kwargs):
    outputs = original_forward(*args, **kwargs)
    forward_call_count[0] += 1
    if forward_call_count[0] <= 20:
        top5 = torch.topk(outputs[0][:, -1, :], 5)
        tokens = [tokenizer.decode([i.item()]) for i in top5.indices[0]]
        print(f"[STEP {forward_call_count[0]}] TOP5 tokens: {tokens}")
    return outputs

lazy_qwen3_model.forward = debug_forward

output_ids = lazy_qwen3_model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=input_ids.shape[1] + 200,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    do_sample=False,
)
lazy_qwen3_model.forward = original_forward 
torch.cuda.synchronize()
t5 = time.time()
output_len_lazy = output_ids.shape[1] - input_ids.shape[1]
print(f"[LazyQwen3] 生成总时间: {(t5-t4)*1000:.1f}ms, 输出tokens: {output_len_lazy}, 平均每token: {(t5-t4)*1000/output_len_lazy:.1f}ms")
print(f"[LazyQwen3] 输出:\n{tokenizer.decode(output_ids[0], skip_special_tokens=True)}")