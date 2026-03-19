import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4" 
from transformers import Qwen3ForCausalLM, AutoTokenizer
from models import *
import torch

# Setting the device to cuda if available, otherwise cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
local_path = "/data/zn/model/models/Qwen3-8B"   # ← 换成你本地的实际路径
qwen3_model = Qwen3ForCausalLM.from_pretrained(local_path, torch_dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained(local_path)

# Mapping the original model to LazyQwen3ForCausalLM
lazy_qwen3_model = LazyQwen3ForCausalLM.from_qwen3_state_dict(
    qwen3_model.state_dict(), 
    qwen3_model.config,
    pruning_rates={i: 0.1 for i in range(32)}, 
).to(device)

# Using the tokenizer to encode the input sequence
prompt = 'Write a delicious recipe for french fries.\n\n'
input_sequence = tokenizer([prompt], return_tensors="pt")

# Generating the output sequence with the original Qwen3 model
output_ids = qwen3_model.generate(input_sequence["input_ids"].to(device), max_length=250, num_return_sequences=1)

# Decoding the output sequence
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Original Qwen3 model output:\n{output_text}")

# Generating the output sequence with the Lazy Qwen3 model
output_ids = lazy_qwen3_model.generate(
    input_sequence["input_ids"].to(device), 
    attention_mask=input_sequence["attention_mask"].to(device), 
    max_length=250, 
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    do_sample=True,
)

# Decoding the output sequence
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Lazy LLaMa 2 model output:\n{output_text}")