from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer, Qwen3RotaryEmbedding,
    apply_rotary_pos_emb, repeat_kv, eager_attention_forward,
)
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

import torch
import os
from context import Context
import torch.nn as nn
from typing import Tuple, Optional
from config import LazyQwen3Config


class DecoderLayer(nn.Module):  # 自定义的 DecoderLayer，继承自 nn.Module
    """ 
    基于 HuggingFace 的 Qwen3DecoderLayer 封装的一层 Decoder，
    额外实现了：
    - KV Cache（缓存 K/V 用于加速 decode）
    - Aux Cache（用于保存被剪枝掉的 token，方便后续补回）
    - 动态 token 剪枝（根据注意力重要性分数，丢弃不重要的上下文 token）
    """

    def __init__(self, config: LazyQwen3Config, layer_idx: int):  # 构造函数，传入扩展配置和当前层编号
        """ 
        初始化 DecoderLayer。

        参数：
            config: LazyQwen3Config，包含模型结构、剪枝比例等超参
            layer_idx: 当前 decoder 层的索引（0-based）
        """
        super().__init__()  # 调用 nn.Module 的初始化
        self.config = config  # 保存配置
        self.layer_idx = layer_idx  # 记录当前是第几层
        # 全局调试开关：是否强制 attention 计算在 FP32 下进行（避免 bf16/FP16 导致 NaN）
        self._force_attn_fp32 = os.getenv("LAZYLLAMA_ATTN_FP32", "0") == "1"  # 从环境变量读取，等于 "1" 则开启
        self._attn_casted = False  # 标记 self_attn 是否已经被转成 float32，一次即可
        # 调试开关：关闭剪枝（包括重要性计算、AuxCache 与 prune），仅保留 KV 路径
        # 1 = 完全不剪枝；0 = 按 config.pruning_rates 正常剪枝
        self._disable_prune = os.getenv("LAZYLLAMA_DISABLE_PRUNE", "0") == "1"  # 开启时，完全禁用 prune 逻辑
        # 调试开关：关闭 HF KV cache，尽量靠近“纯 HF self_attn”行为
        # 1 = 关闭 KV cache（past_key_values=None，不调用 context.update_kv_cache）
        # 0 = 正常 Lazy 路径（默认）
        self._disable_kv_cache = os.getenv("LAZYLLAMA_DISABLE_KV_CACHE", "0") == "1"  # 为 1 时完全不用 KV cache
        # 调试开关：指定若干层在 prefill 阶段直接走原生 Qwen3DecoderLayer.forward，
        # 完全绕过自定义 KV / Aux / 剪枝 / 手写 RoPE 等逻辑，用于逐层二分排查问题。
        # 用法示例：
        #   LAZYLLAMA_LAYER_HF_ONLY="35"      只让第 35 层走 HF 原始实现
        #   LAZYLLAMA_LAYER_HF_ONLY="0,1,35"  让 0、1、35 层走 HF 实现
        #   LAZYLLAMA_LAYER_HF_ONLY="all"     所有层都走 HF，实现纯 HF 路径
        raw = os.getenv("LAZYLLAMA_LAYER_HF_ONLY", "").strip()  # 从环境变量中读出配置字符串
        if raw:  # 如果字符串非空，说明打开了某种 HF-only 调试模式
            if raw.lower() == "all":  # 如果是 "all"，所有层都用 HF 原始实现
                self._hf_debug_layers = "all"  # 用特殊标记 "all" 代表全部层
            else:
                try:
                    layers: set[int] = set()  # 用集合存放被指定的层索引
                    for part in raw.split(","):  # 支持 "0,1,35" 或 "0-3,10" 这类写法
                        part = part.strip()  # 去掉多余空格
                        if not part:  # 空片段跳过
                            continue
                        if "-" in part:  # 支持区间，例如 "10-20"
                            a, b = part.split("-", 1)  # 按第一个 "-" 切开
                            a, b = int(a), int(b)  # 转为整数
                            if a <= b:  # 正序区间
                                layers.update(range(a, b + 1))  # 加入所有层号
                            else:  # 反序区间，做个保护
                                layers.update(range(b, a + 1))  # 反过来也加入
                        else:  # 单个数字，比如 "12"
                            layers.add(int(part))  # 直接添加到集合
                    self._hf_debug_layers = layers  # 保存为 set
                except Exception:  # 解析失败时，直接关闭 HF-only 模式，避免影响正常逻辑
                    self._hf_debug_layers = None
        else:
            self._hf_debug_layers = None  # 没设置环境变量时，不启用 HF-only 调试路径

        # 直接复用上层传入的 config，确保与 RotaryEmbedding 使用的 config 完全一致，
        # 注意：不要在这里再去修改 config.head_dim，
        # 否则会让 Qwen3Attention 的 head_dim 和 Qwen3RotaryEmbedding 的 RoPE 维度不一致，
        # 导致 apply_rotary_pos_emb 中出现 32 vs 128 之类的维度冲突。
        attn_impl = os.getenv("LAZYLLAMA_ATTN_IMPL", "sdpa").lower()
        if attn_impl not in {"sdpa", "eager", "flash_attention_2"}:
            attn_impl = "sdpa"
        self.config._attn_implementation = attn_impl  # 将实现方式写回 config，供 HF 层使用

        # 始终构造底层 Qwen3DecoderLayer，保证参数名中包含 "decoder"，
        # 以便与 from_qwen3_state_dict 里插入的 key 前缀对齐
        self.decoder = Qwen3DecoderLayer(self.config, self.layer_idx)  # 用相同 config 和 layer_idx 初始化 HF 的 Qwen3DecoderLayer
        # 原本用 0 作为所有层的 layer_idx，在我们自定义 HFCache 的实现下功能上也能跑通，
        # 但会导致调试日志里所有层都显示为 layer=0，不利于排查问题。
        # print(f"[DecoderLayer] attn_implementation: {self.decoder.self_attn.config._attn_implementation}")  # 可选调试打印

    def forward(
            self,
            context: Context,  # 封装当前 batch 的 hidden_states / 位置 / KV cache / Aux cache 等信息
            causal_mask: torch.FloatTensor,  # 4 维因果 mask，形状接近 [b, 1, q_len, kv_len]
            rotary_emb: Qwen3RotaryEmbedding,  # HF 的 RoPE 模块，用来计算 cos/sin
            output_attentions: bool,  # 是否输出注意力权重（目前没有真正向外返回）
        ) -> Tuple[Context, Optional[torch.Tensor]]:  # 返回新的 Context，还有可选的注意力张量
        """ 
        执行一层 decoder 的前向计算：
        - 更新 hidden_states
        - 更新 KV Cache / Aux Cache
        - （可选）计算注意力权重（目前内部用于重要性估计）
        """
        # Init KV view early: needed by keys_idxs_to_tokens_idxs (HF debug) and get_aux_cache
        local_kv_cache = context.get_kv_cache(self.layer_idx)
        is_prefill = context.hidden_states.shape[1] > 1

        # === HF debug branch (rarely used, env LAZYLLAMA_LAYER_HF_ONLY) ===
        if (
            is_prefill
            and self._hf_debug_layers is not None
            and (self._hf_debug_layers == "all" or self.layer_idx in self._hf_debug_layers)
        ):
            causal_mask_hf = torch.index_select(causal_mask, 3, context.keys_idxs_to_tokens_idxs)
            causal_mask_hf = torch.index_select(causal_mask_hf, 2, context.hidden_states_idxs)
            causal_mask_hf = causal_mask_hf.to(context.hidden_states.device)
            position_embeddings_hf = rotary_emb(context.hidden_states, context.hidden_states_positions)
            with torch.no_grad():
                hf_outputs = self.decoder(
                    hidden_states=context.hidden_states,
                    attention_mask=causal_mask_hf,
                    position_embeddings=position_embeddings_hf,
                    past_key_values=None,
                    use_cache=False,
                    cache_position=context.hidden_states_idxs,
                )
                hidden_states = hf_outputs[0]
                if hidden_states.dtype != context.hidden_states.dtype:
                    hidden_states = hidden_states.to(context.hidden_states.dtype)
                context.hidden_states = hidden_states
            return (context,)

        # AuxCache: restore tokens pruned by previous layer but needed here
        if self.layer_idx > 0:
            context.get_aux_cache(self.layer_idx)

        # Pre-prune from previous layer's importance scores
        if not self._disable_prune and self.layer_idx > 0:
            context.apply_pre_prune_from_prev_importance(
                self.layer_idx, float(self.config.pruning_rates[self.layer_idx])
            )

        is_prefill = context.hidden_states.shape[1] > 1

        # ---- Construct attention mask ----
        causal_mask = torch.index_select(causal_mask, 3, context.keys_idxs_to_tokens_idxs)
        causal_mask = torch.index_select(causal_mask, 2, context.hidden_states_idxs)
        causal_mask = causal_mask.to(context.hidden_states.device)

        # ---- Position embeddings (RoPE cos/sin) ----
        attn_pos = rotary_emb(context.hidden_states, context.hidden_states_positions)

        # ---- LayerNorm ----
        residual = context.hidden_states
        kv_cache_len_before = context.in_kv_cache_idxs.shape[0]

        with torch.no_grad():
            attn_hs = self.decoder.input_layernorm(context.hidden_states)

        attn_mask = causal_mask

        # ---- Debug: FP32 / mask overrides ----
        mask_mode = os.getenv("LAZYLLAMA_MASK_MODE", "keep").lower()
        if mask_mode == "zero":
            attn_mask = torch.zeros_like(attn_mask)
        if self._force_attn_fp32:
            if not self._attn_casted:
                self.decoder.self_attn.to(torch.float32)
                self._attn_casted = True
            attn_hs = attn_hs.float()
            attn_mask = attn_mask.float()
            attn_pos = rotary_emb(attn_hs, context.hidden_states_positions)
        if mask_mode == "clamp" and attn_mask is not None and attn_mask.dtype in (torch.bfloat16, torch.float16):
            attn_mask = attn_mask.clamp(min=float(os.getenv("LAZYLLAMA_MASK_CLAMP_MIN", "-1e4")))

        # ==================================================================
        # Inlined self_attn: Q/K/V computed ONCE, shared with importance
        # prefill 热点（Triton/CUDA 实验）：q/k/v_proj、apply_rotary_pos_emb、下方 attn_fn 调用；
        # 剪枝关闭时可对比 SDPA/Flash；开启时另有 importance 的 q_last@K 小矩阵乘。
        # ==================================================================
        with torch.no_grad():
            sa = self.decoder.self_attn
            input_shape = attn_hs.shape[:-1]
            hidden_shape = (*input_shape, -1, sa.head_dim)

            # Q / K / V projections -- done ONCE
            query_states = sa.q_norm(sa.q_proj(attn_hs).view(hidden_shape)).transpose(1, 2)
            key_states = sa.k_norm(sa.k_proj(attn_hs).view(hidden_shape)).transpose(1, 2)
            value_states = sa.v_proj(attn_hs).view(hidden_shape).transpose(1, 2)

            # RoPE -- done ONCE
            cos, sin = attn_pos
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # ---- Importance extraction (cheap: single query x all keys) ----
            if not self._disable_prune:
                k_for_imp = repeat_kv(key_states, sa.num_key_value_groups)
                last_idx = context.tkns_idxs_to_hidden_states_idxs[context.sequence_length - 1]
                q_last = query_states[:, :, last_idx:last_idx + 1, :]
                imp = torch.softmax(
                    (q_last @ k_for_imp.transpose(-2, -1)) * sa.scaling, dim=-1
                )
                scores = imp[0, :, 0, :].mean(dim=0).detach().clone()
                hs_idxs = context.hidden_states_idxs
                pos_last = (hs_idxs == context.sequence_length - 1).nonzero(as_tuple=True)[0]
                if pos_last.numel() > 0:
                    scores[pos_last[0]] = float("inf")
                context.token_importance_prev[hs_idxs.long()] = scores.float()
                del k_for_imp, q_last, imp, scores

            # ---- KV cache update ----
            if not self._disable_kv_cache:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": context.hidden_states_idxs}
                key_states, value_states = local_kv_cache.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )

            # ---- Attention kernel (reuses Q/K/V already computed) ----
            if attn_mask is not None and attn_mask.dtype != query_states.dtype:
                attn_mask = attn_mask.to(query_states.dtype)
            attn_fn = ALL_ATTENTION_FUNCTIONS.get_interface(
                sa.config._attn_implementation, eager_attention_forward
            )
            attn_output, _ = attn_fn(
                sa, query_states, key_states, value_states, attn_mask,
                dropout=0.0, scaling=sa.scaling, sliding_window=sa.sliding_window,
            )

            hidden_states = sa.o_proj(attn_output.reshape(*input_shape, -1).contiguous())

        # ---- dtype alignment ----
        if hidden_states.dtype != context.hidden_states.dtype:
            hidden_states = hidden_states.to(context.hidden_states.dtype)

        # ---- Residual + MLP ----
        with torch.no_grad():
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.decoder.post_attention_layernorm(hidden_states)
            mlp_dtype = self.decoder.mlp.gate_proj.weight.dtype
            if hidden_states.dtype != mlp_dtype:
                hidden_states = hidden_states.to(mlp_dtype)
            hidden_states = self.decoder.mlp(hidden_states)
            if residual.dtype != hidden_states.dtype:
                residual = residual.to(hidden_states.dtype)
            hidden_states = residual + hidden_states

        if hidden_states.isnan().any():
            print(f"[NaN] layer={self.layer_idx}, NaN appeared after MLP")
        context.hidden_states = hidden_states

        # ---- Write-back KV cache status bits ----
        if not self._disable_kv_cache:
            context.update_kv_cache(local_kv_cache, self.layer_idx, kv_cache_offset=kv_cache_len_before)

        return (context,)