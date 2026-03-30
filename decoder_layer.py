from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding  # 从 transformers 中导入 Qwen3 的解码层和 RoPE 模块
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config  # 导入 Qwen3 的配置类（这里主要是类型引用）

import torch  # PyTorch 主包
import os  # 用于读取环境变量，控制调试开关
from context import Context  # 项目自定义的 Context 类，封装 hidden_states、KV cache 等
import torch.nn as nn  # 神经网络模块
from typing import Tuple, Optional  # 类型注解工具
from config import LazyQwen3Config  # 项目自定义的 Qwen3 配置，扩展了 HF 的配置
decode_step_count = [0]  # 一个列表包装的计数器（目前在本文件中没使用，可能用于外部调试）


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
        attn_impl = os.getenv("LAZYLLAMA_ATTN_IMPL", "eager").lower()  # 读取注意力实现方式（sdpa/eager/flash_attention_2）
        if attn_impl not in {"sdpa", "eager", "flash_attention_2"}:  # 如果不在允许列表里
            attn_impl = "sdpa"  # 默认退回到 sdpa
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
        # 始终初始化当前层的 KV cache / 索引（Context 内部依赖这些索引），
        # 真正是否在 attention 中使用取决于 self._disable_kv_cache。
        local_kv_cache = context.get_kv_cache(self.layer_idx)  # 从 Context 中取出/初始化本层的 KV cache 视图
        is_prefill = context.hidden_states.shape[1] > 1  # 如果当前序列长度 > 1，视为 prefill 阶段，否则视为 decode 单 token 阶段

        # === 层级 HF 调试分支（仅 prefill 生效） ===
        # 尽量在访问 Aux Cache 之前就决定是否走纯 HF 路径，
        # 避免 HF-only 模式下还去拼接 Aux Cache 出现维度不一致等问题。
        if (
            is_prefill  # 只在 prefill 时允许 HF-only 分支
            and self._hf_debug_layers is not None  # 配置里启用了 HF-only 调试
            and (self._hf_debug_layers == "all" or self.layer_idx in self._hf_debug_layers)  # 当前层需要走 HF-only
        ):
            # 为 HF 路径构造与当前 Context 对应的 causal_mask
            causal_mask_hf = torch.index_select(causal_mask, 3, context.keys_idxs_to_tokens_idxs)  # 先按 keys_idxs_to_tokens_idxs 选择 KV 维度
            if is_prefill:  # prefill 还需要在 query 维度上做一次 index_select
                causal_mask_hf = torch.index_select(causal_mask_hf, 2, context.hidden_states_idxs)  # 按 hidden_states_idxs 选择 query 位置
            causal_mask_hf = causal_mask_hf.to(context.hidden_states.device)  # 把 mask 放到和 hidden_states 相同的 device

            # 完全仿照 HF Qwen3Model：用同一个 RotaryEmbedding 计算 (cos, sin)
            position_embeddings_hf = rotary_emb(context.hidden_states, context.hidden_states_positions)  # 调用 HF 的 RoPE 模块，返回 (cos, sin)
            try:
                cos_hf, sin_hf = position_embeddings_hf  # 拆出 cos 和 sin
                # if self.layer_idx == 0 and is_prefill:  # 只在第 0 层 prefill 做详细打印，避免刷屏
                #     print(
                #         f"[HF-only RoPE] cfg.head_dim={getattr(self.config, 'head_dim', None)}, "
                #         f"attn.head_dim={getattr(self.decoder.self_attn, 'head_dim', None)}, "
                #         f"cos.shape={cos_hf.shape}, sin.shape={sin_hf.shape}"
                #     )  # 打印 RoPE 相关维度信息，帮助排查维度不一致问题
            except Exception:  # 保护性 try-catch，防止打印本身出错
                pass

            if self.layer_idx == 0 and is_prefill:  # 第 0 层额外打印 self_attn 的基本结构
                sa = self.decoder.self_attn  # 取出 HF 内部的 self_attn 模块
                # print(
                #     f"[HF-only self_attn] hidden_size={sa.config.hidden_size}, "
                #     f"num_heads={sa.config.num_attention_heads}, "
                #     f"head_dim={sa.head_dim}"
                # )  # 输出头数、hidden_size、head_dim 等

            with torch.no_grad():  # HF-only 调试时，不参与梯度
                hf_outputs = self.decoder(
                    hidden_states=context.hidden_states,  # 直接把 Context 里的 hidden_states 喂给 HF 层
                    attention_mask=causal_mask_hf,  # 传入刚刚对齐过的 HF 风格 attention_mask
                    position_embeddings=position_embeddings_hf,  # 传入 HF RoPE 的 (cos, sin)
                    # 只在 prefill 阶段做 HF 对齐，不使用 past_key_values
                    past_key_values=None,  # 不传 KV cache，强制 HF 自己算
                    use_cache=False,  # 不要求 HF 返回新的 cache
                    cache_position=context.hidden_states_idxs,  # 传入当前 query 的索引位置
                )
                hidden_states = hf_outputs[0]  # HF Qwen3DecoderLayer 的第一个返回值是新的 hidden_states
                if hidden_states.dtype != context.hidden_states.dtype:  # 如果精度和 Context 不同（比如 fp32 vs bf16）
                    hidden_states = hidden_states.to(context.hidden_states.dtype)  # 转成和 Context 一样的 dtype
                context.hidden_states = hidden_states  # 写回 Context
            # 在纯 HF 调试路径下，我们不更新 KV / Aux / 剪枝，直接返回，只更新 hidden_states
            return (context,)  # 按主流程约定，仍然返回一个只包含 Context 的元组

        # AuxCache 合并只应该发生在 prefill：decode 阶段 query_len=1，
        # 若把 aux tokens 拼到 hidden_states，会导致 causal_mask 的 query 维度(=1)与 hidden_states_idxs(含历史 token)不匹配。
        if is_prefill and self.layer_idx > 0:  # 第 0 层之前没有 Aux cache，只有从第 1 层开始才会用到
            context.get_aux_cache(self.layer_idx)  # 确保本层的 Aux cache 已初始化
        if self.layer_idx == 0 and context.hidden_states.shape[1] > 1:  # 只在第 0 层的 prefill 打印输入统计
            hs = context.hidden_states  # 取出当前 hidden_states
            # print(
            #     f"[layer0 input] mean={hs.mean().item():.4f}, std={hs.std().item():.4f}, max={hs.max().item():.4f}"
            # )  # 打印均值、方差、最大值，排查数值异常

        # RoPE 位置编码输入检查：只看前几层避免刷屏
        if self.layer_idx == 0:  # 专门在第 0 层检查 RoPE 的 position 输入
            pos = context.hidden_states_positions  # 取出每个 token 的 position index
            # print(
            #     f"[Layer0 RoPE Check] positions shape={pos.shape}, first5={pos[0, :5].tolist()}, last5={pos[0, -5:].tolist()}"
            # )  # 打印位置向量的形状以及前后若干个值

        # ---- Safety check: surface out-of-range indices before CUDA assert ----
        # When pruning is enabled, hidden_states_idxs / keys_idxs_to_tokens_idxs may become sparse.
        # If any index is outside the causal_mask dimensions, CUDA will trigger a device-side assert
        # that is hard to debug. Convert that into a readable Python exception.
        if is_prefill:
            try:
                q_dim = int(causal_mask.size(2))
                kv_dim = int(causal_mask.size(3))
                hs_idxs = context.hidden_states_idxs
                key_idxs = context.keys_idxs_to_tokens_idxs

                if hs_idxs.numel() > 0:
                    hs_min = int(hs_idxs.min().item())
                    hs_max = int(hs_idxs.max().item())
                    if hs_min < 0 or hs_max >= q_dim:
                        head = hs_idxs[:8].detach().cpu().tolist()
                        tail = hs_idxs[-8:].detach().cpu().tolist() if hs_idxs.numel() > 8 else []
                        raise RuntimeError(
                            "[Lazy-Llama] hidden_states_idxs out of range for causal_mask query dim. "
                            f"layer={self.layer_idx}, causal_mask.shape={tuple(causal_mask.shape)}, q_dim={q_dim}, "
                            f"hs_idxs.numel={int(hs_idxs.numel())}, hs_min={hs_min}, hs_max={hs_max}, head={head}, tail={tail}"
                        )

                if key_idxs is not None and key_idxs.numel() > 0:
                    k_min = int(key_idxs.min().item())
                    k_max = int(key_idxs.max().item())
                    if k_min < 0 or k_max >= kv_dim:
                        head = key_idxs[:8].detach().cpu().tolist()
                        tail = key_idxs[-8:].detach().cpu().tolist() if key_idxs.numel() > 8 else []
                        raise RuntimeError(
                            "[Lazy-Llama] keys_idxs_to_tokens_idxs out of range for causal_mask key/value dim. "
                            f"layer={self.layer_idx}, causal_mask.shape={tuple(causal_mask.shape)}, kv_dim={kv_dim}, "
                            f"key_idxs.numel={int(key_idxs.numel())}, k_min={k_min}, k_max={k_max}, head={head}, tail={tail}"
                        )
            except RuntimeError:
                raise
            except Exception:
                # Never let debug checks break normal execution.
                pass

        causal_mask = torch.index_select(causal_mask, 3, context.keys_idxs_to_tokens_idxs)  # 先按 KV 侧 index_select，将 4D mask 的最后一维对齐到真实 KV 序列
        # decode 阶段 hidden_states 只有 1 个 token，dim=1 已经是 1，不需要 index_select；
        # prefill 阶段才需要按 hidden_states_idxs 筛选 query 维度
        if is_prefill:  # prefill 时序列长度 > 1
            causal_mask = torch.index_select(causal_mask, 2, context.hidden_states_idxs)  # 在 query 维度用 hidden_states_idxs 做筛选
        causal_mask = causal_mask.to(context.hidden_states.device)  # 把 mask 移到 hidden_states 所在的设备

        if self.layer_idx == 0 and context.hidden_states.shape[1] > 1:  # 第 0 层 prefill 再次打印位置索引信息
            pos = context.hidden_states_positions  # 位置索引张量
            # print(
            #     f"[layer0 positions] shape={pos.shape}, first5={pos[0,:5].tolist()}, last5={pos[0,-5:].tolist()}"
            # )  # 再确认一次位置是否和期望一致
        position_embeddings = rotary_emb(  # 使用 HF 的 RoPE 模块计算 position embeddings（cos, sin）
            context.hidden_states,  # 传入当前 hidden_states（只用 shape）
            context.hidden_states_positions,  # 传入当前 token 的 position indices
        )

        residual = context.hidden_states  # 残差分支，后面 attention 输出会加在它上面
        with torch.no_grad():  # layernorm 本身不需要参与剪枝重要性梯度，这里直接 no_grad 更安全
            hidden_states_norm = self.decoder.input_layernorm(context.hidden_states)  # 调用 HF 的 input_layernorm 做归一化
        kv_cache_len_before = context.in_kv_cache_idxs.shape[0]  # 当前 KV cache 中已经存在的 token 数量，用于后续剪枝偏移
        # causal_mask = causal_mask.float()  # 保留的注释行，之前可能用于强制转换为 float
        # if self.layer_idx == 0 and context.hidden_states.shape[1] > 1:  # 第 0 层 prefill 打印 mask / norm 的统计
        #     print(f"[mask values] min={causal_mask.min():.1f}, max={causal_mask.max():.1f}")  # 打印 mask 的取值范围
        #     print(f"[mask dtype] {causal_mask.dtype}")  # 打印 mask 的 dtype
        #     print(
        #         f"[hidden_states_norm] min={hidden_states_norm.min():.4f}, max={hidden_states_norm.max():.4f}, "
        #         f"has_nan={hidden_states_norm.isnan().any()}"
        #     )  # 检查归一化后的 hidden 是否有 NaN
        # if context.hidden_states.shape[1] > 1 and self.layer_idx <= 2:  # 对前几层 prefill 打印更加详细的形状信息
        #     print(f"[layer{self.layer_idx} pre-attn] hidden_states.shape={context.hidden_states.shape}")  # 打印 hidden_states 形状
        #     print(
        #         f"[layer{self.layer_idx} pre-attn] hidden_states_idxs: first5={context.hidden_states_idxs[:5].tolist()}, "
        #         f"last5={context.hidden_states_idxs[-5:].tolist()}"
        #     )  # 打印 query 端索引映射的前后几项
        #     print(f"[layer{self.layer_idx} pre-attn] in_kv_cache_idxs.shape={context.in_kv_cache_idxs.shape}")  # KV cache 索引形状
        #     print(f"[layer{self.layer_idx} pre-attn] causal_mask.shape={causal_mask.shape}")  # mask 形状
        # SDPA 要求 attention_mask 与 query dtype 一致；在 bf16 下，为了避免 -inf 溢出引起 NaN，后面会做 clamp
        attn_hs = hidden_states_norm  # 即将送入 self_attn 的输入张量（已经 layernorm 过）
        attn_mask = causal_mask  # 注意力 mask（之后可能根据策略修改）
        attn_pos = position_embeddings  # RoPE 的 (cos, sin) 对
        # 读取 mask 模式：
        # - keep: 保持上游构造的 causal mask（用于对齐/正常推理，默认）
        # - clamp: 低精度下对 mask 做下界截断，避免 -inf 溢出
        # - zero: 将 mask 清零（仅用于调试，会破坏因果约束）
        mask_mode = os.getenv("LAZYLLAMA_MASK_MODE", "keep").lower()
        clamp_min = float(os.getenv("LAZYLLAMA_MASK_CLAMP_MIN", "-1e4"))  # 如果是 clamp 模式，最小 clamp 值
        '''
        if self.layer_idx == 0 and context.hidden_states.shape[1] > 1:  # 第 0 层 prefill 再次检查 mask / hs 数值
            # print(
            #     f"[layer0 prefill attn_mask] shape={attn_mask.shape}, min={attn_mask.min():.1f}, "
            #     f"max={attn_mask.max():.1f}, has_nan={attn_mask.isnan().any()}, has_inf={attn_mask.isinf().any()}"
            # )  # 打印 mask 数值统计
            # print(
            #     f"[layer0 prefill attn_hs] shape={attn_hs.shape}, has_nan={attn_hs.isnan().any()}, "
            #     f"has_inf={attn_hs.isinf().any()}"
            # )  # 打印 attn_hs 是否有 NaN / Inf
            # 额外检查 RoPE cos/sin 数值范围
            try:
                cos, sin = attn_pos  # RoPE 输出的 cos 和 sin
                # print(
                #     f"[layer0 prefill RoPE] cos min={cos.min().item():.4f}, max={cos.max().item():.4f}, "
                #     f"has_nan={cos.isnan().any()}, has_inf={cos.isinf().any()}"
                # )  # 检查 cos 是否数值正常
                # print(
                #     f"[layer0 prefill RoPE] sin min={sin.min().item():.4f}, max={sin.max().item():.4f}, "
                #     f"has_nan={sin.isnan().any()}, has_inf={sin.isinf().any()}"
                # )  # 检查 sin 是否数值正常
            except Exception:  # 保护性异常捕获
                pass
        '''
        # layer0 专用调试开关：更激进的 FP32 / mask 实验
        if self.layer_idx == 0:  # 仅对第 0 层读取以下两个额外调试环境变量
            layer0_force_fp32 = os.getenv("LAZYLLAMA_LAYER0_FP32", "0") == "1"  # 是否强制第 0 层在 FP32 下跑 attention
            layer0_no_mask = os.getenv("LAZYLLAMA_LAYER0_NO_MASK", "0") == "1"  # 是否在第 0 层完全不用外部 mask
        else:
            layer0_force_fp32 = False  # 非第 0 层忽略这些开关
            layer0_no_mask = False
        if mask_mode == "zero":  # 如果全局 mask_mode 设为 "zero"，则把 mask 直接清零
            attn_mask = torch.zeros_like(attn_mask)  # 构造与原 mask 同形状的全 0 tensor
        if self._force_attn_fp32:  # 全局开关：强制所有层 attention 逻辑用 FP32 计算
            if not self._attn_casted:  # 只在第一次真正执行时，将 self_attn 权重 cast 到 float32
                self.decoder.self_attn.to(torch.float32)  # 把自注意力模块迁移到 FP32
                self._attn_casted = True  # 标记已经 cast 过
            attn_hs = attn_hs.float()  # 输入张量也转成 FP32
            attn_mask = attn_mask.float()  # mask 也转成 FP32，方便与 attn_hs 匹配
            # 重新用 fp32 hidden_states 计算 rotary，避免低精度导致 NaN
            attn_pos = rotary_emb(attn_hs, context.hidden_states_positions)  # 用 FP32 版本的 hidden_states 重新计算 RoPE
        # 仅在第 0 层启用的更激进 FP32 / mask 实验
        if layer0_force_fp32:  # 如果第 0 层强制 fp32
            attn_hs = attn_hs.float()  # 再次确保 attn_hs 是 float32
            if attn_mask is not None:  # 如果 mask 不是 None
                attn_mask = attn_mask.float()  # 同样转为 float32
            attn_pos = rotary_emb(attn_hs, context.hidden_states_positions)  # 再次用 FP32 hidden_states 算 RoPE
            # print(
            #     f"[layer0 fp32 debug] attn_hs dtype={attn_hs.dtype}, "
            #     f"attn_mask dtype={(attn_mask.dtype if attn_mask is not None else None)}"
            # )  # 打印 dtype 方便调试
        if layer0_no_mask:  # 如果第 0 层不用外部 mask
            attn_mask = None  # 直接把 attention_mask 置为 None，依赖 HF 内部的 causal mask
            print("[layer0 fp32 debug] attention_mask overridden to None (use internal causal mask)")  # 提示信息
        # if attn_mask.dtype != attn_hs.dtype:
        #     attn_mask = attn_mask.to(attn_hs.dtype)
        # 避免 -inf 在 bf16 下溢出导致 NaN
        if attn_mask is not None and mask_mode == "clamp" and attn_mask.dtype in (torch.bfloat16, torch.float16):  # 如果是低精度 + clamp 模式
            attn_mask = attn_mask.clamp(min=clamp_min)  # 将 mask 下界截断到 clamp_min，避免 -inf 溢出

        # [ surgically replace layer 0 prefill RoPE ]
        '''
        if self.layer_idx == 0 and is_prefill:  # 对第 0 层 prefill，手写一份 HF attention 逻辑，重点是 RoPE 的使用
            # Manually implement the HF attention logic for layer 0 to ensure correct RoPE application
            with torch.no_grad():  # 这里主要用于数值对齐调试，不计算梯度
                # 1. Project Q, K, V
                q = self.decoder.self_attn.q_proj(attn_hs)  # 使用 HF 的 q_proj 线性层得到 query 投影
                k = self.decoder.self_attn.k_proj(attn_hs)  # 使用 HF 的 k_proj 得到 key 投影
                v = self.decoder.self_attn.v_proj(attn_hs)  # 使用 HF 的 v_proj 得到 value 投影

                # 2. Reshape for attention heads
                batch_size, seq_len, _ = attn_hs.shape  # 记录 batch 和序列长度
                q = q.view(  # 把 q reshape 成 [b, num_heads, seq_len, head_dim]
                    batch_size,
                    seq_len,
                    self.config.num_attention_heads,
                    self.config.head_dim,
                ).transpose(1, 2)  # 交换维度得到 [b, h, s, d]
                k = k.view(  # 把 k reshape 成 [b, num_key_value_heads, seq_len, head_dim]
                    batch_size,
                    seq_len,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                ).transpose(1, 2)  # 交换后 [b, kv_heads, s, d]
                v = v.view(  # 把 v 也 reshape 成 [b, num_key_value_heads, seq_len, head_dim]
                    batch_size,
                    seq_len,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                ).transpose(1, 2)  # 最后形状同上

                # 3. Apply RoPE (the critical part)
                # 原始 HF 实现会在内部 apply_rotary_pos_emb，这里我们显式地做一次，方便调试维度 / 数值。
                cos, sin = rotary_emb(v, context.hidden_states_positions)  # 利用 v 的 shape 作为模板，只关心 batch/seq/head_dim
                print(f"[RoPE] max_seq_len_cached={rotary_emb.max_seq_len_cached}, positions_max={context.hidden_states_positions.max()}")
                # Debug print to check if RoPE values are sane
                print(
                    f"[layer0 prefill RoPE REPLACEMENT] cos min={cos.min().item():.4f}, "
                    f"max={cos.max().item():.4f}, has_nan={cos.isnan().any()}, has_inf={cos.isinf().any()}"
                )  # 打印 cos 的数值范围
                print(
                    f"[layer0 prefill RoPE REPLACEMENT] sin min={sin.min().item():.4f}, "
                    f"max={sin.max().item():.4f}, has_nan={sin.isnan().any()}, has_inf={sin.isinf().any()}"
                )  # 打印 sin 的数值范围

                from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb  # 导入 HF 的 RoPE 应用函数
                q, k = apply_rotary_pos_emb(q, k, cos, sin)  # 对 q / k 应用旋转位置编码，得到带位置信息的向量

                # 4. Repeat K, V for GQA
                from transformers.models.qwen3.modeling_qwen3 import repeat_kv  # 导入 KV 重复函数（GQA）
                k = repeat_kv(  # 将 k 的 kv 头重复成和注意力头数相同
                    k,
                    self.config.num_attention_heads // self.config.num_key_value_heads,
                )
                v = repeat_kv(  # 将 v 的 kv 头重复成和注意力头数相同
                    v,
                    self.config.num_attention_heads // self.config.num_key_value_heads,
                )

                # 5. Standard attention calculation
                attn_weights = torch.matmul(  # 计算注意力打分矩阵 [b, h, q_len, kv_len]
                    q,
                    k.transpose(2, 3),  # 把 k 的最后两个维度交换
                ) / (self.config.head_dim ** 0.5)  # 除以 sqrt(d) 做缩放
                if attn_mask is not None:  # 如果有外部传入的 mask
                    attn_weights = attn_weights + attn_mask  # 将 mask 加到打分上（通常是大负值）

                attn_weights = nn.functional.softmax(  # 在最后一个维度上做 softmax，得到注意力分布
                    attn_weights, dim=-1, dtype=torch.float32
                ).to(q.dtype)  # 计算时用 FP32，再 cast 回 q 的 dtype
                attn_output = torch.matmul(attn_weights, v)  # [b, h, q_len, kv_len] @ [b, h, kv_len, d] -> [b, h, q_len, d]

                # 6. Reshape output
                attn_output = attn_output.transpose(1, 2).contiguous()  # 先从 [b, h, s, d] 转成 [b, s, h, d]
                attn_output = attn_output.reshape(  # 再 view 成 [b, s, hidden_size]
                    batch_size,
                    seq_len,
                    self.config.hidden_size,
                )

                # 7. Final projection
                hidden_states = self.decoder.self_attn.o_proj(attn_output)  # 通过输出投影 o_proj 映射回 hidden_size
        else:  # 非第 0 层，或者 decode 阶段，直接走 HF 的 self_attn 接口
        '''
        with torch.no_grad():  # 为了对齐 / 调试，这里不计算梯度
            hidden_states, _ = self.decoder.self_attn(
                hidden_states=attn_hs,  # 传入 layernorm 之后的 hidden_states
                attention_mask=attn_mask,  # 传入对齐后的 mask
                position_embeddings=attn_pos,  # 传入计算好的 RoPE (cos, sin)
                past_key_values=None if self._disable_kv_cache else local_kv_cache,  # 如果关闭 KV cache，就不传 past_key_values
                cache_position=None if self._disable_kv_cache else context.hidden_states_idxs,  # 当前 query 在 cache 中的位置
                output_attentions=False,  # 不让 HF 返回注意力权重（我们自己在下面算一份重要性）
            )

        if hidden_states.dtype != context.hidden_states.dtype:  # 如果 self_attn 输出的 dtype 和 Context 不一致
            hidden_states = hidden_states.to(context.hidden_states.dtype)  # 统一 cast 成 Context 的 dtype
        # if self.layer_idx <= 2 and context.hidden_states.shape[1] > 1:  # 前几层 prefill 打印 attention 输出统计
        #     print(f"[layer{self.layer_idx}] attn_out.shape={hidden_states.shape}")  # 打印形状
        #     print(
        #         f"[layer{self.layer_idx}] attn_out: mean={hidden_states.mean():.4f}, std={hidden_states.std():.4f}, "
        #         f"max={hidden_states.abs().max():.4f}"
        #     )  # 打印均值、方差和绝对值最大值
        if context.hidden_states.shape[1] > 1:  # prefill 阶段额外打印最后一个 token 的前 5 维值
            last_hs = hidden_states[0, -1, :5].tolist()  # 取出 batch 中第一个样本的最后一个 token 的前 5 维
            # print(f"[lazy layer{self.layer_idx} last token] {last_hs}")  # 打印出来检查数值稳定性
            if hidden_states.isnan().any():  # 如果 attention 输出里已经有 NaN
                print(f"[NaN] layer={self.layer_idx}, NaN appeared after self_attn")  # 提示 NaN 出现位置
            elif (residual + hidden_states).isnan().any():  # 或者在加残差之后才出现 NaN
                print(f"[NaN] layer={self.layer_idx}, NaN appeared after residual+hidden")  # 打印相应提示
        # ✅ prefill 阶段用 Q/K 计算重要性分数，用于剪枝
        if is_prefill and not self._disable_prune:  # 只有在 prefill 且未禁用 prune 时才计算重要性
            with torch.no_grad():  # 重要性分数只用于决策剪枝，不参与梯度
                q = self.decoder.self_attn.q_proj(attn_hs)  # 再次用 q_proj 计算 query（此处单独用于重要性估计）
                k = self.decoder.self_attn.k_proj(attn_hs)  # 再算一遍 key
                # if self.layer_idx == 0:  # 第 0 层打印 Q/K 的数值统计
                #     print(
                #         f"[layer0 q_proj] min={q.min().item():.4f}, max={q.max().item():.4f}, "
                #         f"std={q.std().item():.4f}, has_nan={q.isnan().any()}, has_inf={q.isinf().any()}"
                #     )
                #     print(
                #         f"[layer0 k_proj] min={k.min().item():.4f}, max={k.max().item():.4f}, "
                #         f"std={k.std().item():.4f}, has_nan={k.isnan().any()}, has_inf={k.isinf().any()}"
                #     )
                batch, seq_len, _ = q.shape  # 取出 batch 和序列长度
                q = q.view(  # 将 q reshape 成 [b, num_heads, s, head_dim]
                    batch,
                    seq_len,
                    self.config.num_attention_heads,
                    self.config.head_dim,
                ).transpose(1, 2)  # 转置后形状为 [b, h, s, d]
                k = k.view(  # 将 k reshape 成 [b, num_key_value_heads, s, head_dim]
                    batch,
                    seq_len,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                ).transpose(1, 2)  # 变成 [b, kv_heads, s, d]
                cos, sin = attn_pos  # 直接复用前面算好的 RoPE (cos, sin)
                from transformers.models.qwen3.modeling_qwen3 import (  # 导入 HF 的 RoPE 和 KV 重复函数
                    apply_rotary_pos_emb,
                    repeat_kv,
                )
                q, k = apply_rotary_pos_emb(q, k, cos, sin)  # 对 q/k 再应用一次 RoPE，保证和 attention 内部逻辑完全一致
                # if self.layer_idx == 0:  # 再次打印 RoPE 之后的 Q/K 统计
                #     print(
                #         f"[layer0 after RoPE] q min={q.min().item():.4f}, max={q.max().item():.4f}, "
                #         f"std={q.std().item():.4f}, has_nan={q.isnan().any()}, has_inf={q.isinf().any()}"
                #     )
                #     print(
                #         f"[layer0 after RoPE] k min={k.min().item():.4f}, max={k.max().item():.4f}, "
                #         f"std={k.std().item():.4f}, has_nan={k.isnan().any()}, has_inf={k.isinf().any()}"
                #     )
                k = repeat_kv(  # 将 key 的 kv 头重复为与注意力头数一致
                    k,
                    self.config.num_attention_heads // self.config.num_key_value_heads,
                )
                last_token_query_idx = context.tkns_idxs_to_hidden_states_idxs[  # 查出“最后一个 token”对应在 hidden_states 里的索引
                    context.sequence_length - 1
                ]
                q_last = q[:, :, last_token_query_idx:last_token_query_idx + 1, :]  # 只保留最后一个 token 的 query，形状 [b, h, 1, d]
                scale = self.config.head_dim ** -0.5  # 缩放因子 1/sqrt(d)
                attn_scores = torch.matmul(  # 与 k 做点积，得到 [b, h, 1, kv_len]
                    q_last, k.transpose(-2, -1)
                ) * scale  # 乘以缩放因子
                # if self.layer_idx == 0:  # 打印 raw attention scores 的统计
                #     print(
                #         f"[layer0 attn_scores] min={attn_scores.min().item():.4f}, max={attn_scores.max().item():.4f}, "
                #         f"std={attn_scores.std().item():.4f}, has_nan={attn_scores.isnan().any()}, has_inf={attn_scores.isinf().any()}"
                #     )
                attn_scores = torch.softmax(attn_scores, dim=-1)  # 对最后一维做 softmax 得到注意力分布
                # if self.layer_idx == 0:  # 再检查 softmax 后的注意力分布
                #     print(
                #         f"[layer0 attn_probs] min={attn_scores.min().item():.4f}, max={attn_scores.max().item():.4f}, "
                #         f"std={attn_scores.std().item():.4f}, has_nan={attn_scores.isnan().any()}, has_inf={attn_scores.isinf().any()}"
                #     )
                importance_scores_list = attn_scores[0, :, 0, :].mean(dim=0).detach().clone()  # 对所有头取平均，得到每个 token 的重要性分数（长度 = kv_len）
                del q, k, q_last, attn_scores, cos, sin  # 显式释放中间张量，减小显存压力

        with torch.no_grad():  # 后续 MLP 等也不需要梯度（主要用于推理和调试）
            hidden_states = residual + hidden_states  # 加上 attention 的残差
            residual = hidden_states  # 更新 residual，供 MLP 使用
            hidden_states = self.decoder.post_attention_layernorm(hidden_states)  # 调用 HF 的 post_attention_layernorm 做归一化
            mlp_dtype = self.decoder.mlp.gate_proj.weight.dtype  # 取出 MLP 当前权重的 dtype
            if hidden_states.dtype != mlp_dtype:  # 如果 hidden_states 的 dtype 和 MLP 不一致
                hidden_states = hidden_states.to(mlp_dtype)  # 转成与 MLP 权重相同的 dtype（比如 fp16/bf16）
            hidden_states = self.decoder.mlp(hidden_states)  # 通过 HF 的 MLP 子层
            if residual.dtype != hidden_states.dtype:  # 再次检查 residual 的 dtype 是否和 MLP 输出一致
                residual = residual.to(hidden_states.dtype)  # 不一致就做一次 cast
            hidden_states = residual + hidden_states  # 最后再加上 MLP 的残差
        # if self.layer_idx == 0 and context.hidden_states.shape[1] > 1:  # 第 0 层 prefill 打印 MLP 之后的输出统计
        #     print(
        #         f"[layer0 output] mean={hidden_states.mean().item():.4f}, std={hidden_states.std().item():.4f}, "
        #         f"max={hidden_states.max().item():.4f}"
        #     )
        # if self.layer_idx == 35 and context.hidden_states.shape[1] > 1:  # 第 35 层（通常是最后一层）也打印统计信息
        #     print(
        #         f"[layer35 output] mean={hidden_states.mean().item():.4f}, std={hidden_states.std().item():.4f}, "
        #         f"max={hidden_states.max().item():.4f}"
        #     )

        if context.hidden_states.shape[1] > 1 and hidden_states.isnan().any():  # 如果 prefill 阶段 MLP 输出里有 NaN
            print(f"[NaN] layer={self.layer_idx}, NaN appeared after MLP")  # 打印定位信息
        context.hidden_states = hidden_states  # 把本层最终的 hidden_states 写回 Context
        # 纯 HF self_attn 调试模式下不更新 KV cache
        if not self._disable_kv_cache:  # 如果没有关闭 KV cache
            context.update_kv_cache(  # 更新 Context 里的 KV cache 结构
                local_kv_cache,
                self.layer_idx,
                kv_cache_offset=kv_cache_len_before,  # 告诉 Context：本次新增 token 的起始位置
            )

        # 剪枝（可通过 LAZYLLAMA_DISABLE_PRUNE 完全关闭）
        if is_prefill and not self._disable_prune:  # 只有在 prefill 且未禁用 prune 时才真正剪枝
            last_token_key_idx = last_token_query_idx  # 最后一个 token 在 key 侧的索引（与 query 侧一致）
            pruning_rate = self.config.pruning_rates[self.layer_idx]  # 读取本层对应的剪枝率，例如 0.5 表示剪掉 50% token
            if importance_scores_list.shape[0] > 1:  # importance_scores_list 至少要有两个元素才有意义
                importance_scores_list[last_token_key_idx] = float("inf")  # 强制保留最后一个 token（设为 +inf）
                _, to_prune_list_idxs = torch.topk(  # 选出重要性最低的若干 token 作为剪枝目标
                    importance_scores_list,
                    int(pruning_rate * importance_scores_list.shape[0]),  # 要剪枝的元素个数 = 比例 * 总长度
                    largest=False,  # 取最小的 k 个
                )
                to_prune_list_idxs = to_prune_list_idxs + kv_cache_len_before  # 把这些 index 偏移到全局 KV cache 索引空间
            else:  # importance 列表只有一个元素（理论上只有一个 token 时不会剪枝）
                to_prune_list_idxs = torch.tensor(  # 直接创建一个空的 long tensor 表示不剪枝
                    [],
                    dtype=torch.long,
                    device=context.device,
                )
        else:  # 非 prefill 或者禁用剪枝情况
            to_prune_list_idxs = torch.tensor(  # 直接一个空列表，表示本层不剪枝
                [],
                dtype=torch.long,
                device=context.device,
            )

        if to_prune_list_idxs.numel() > 0:  # 如果有需要剪枝的 KV cache index
            to_prune_idxs = context.keys_idxs_to_tokens_idxs[to_prune_list_idxs]  # 映射到“原始 token 索引”空间
        else:  # 如果没有要剪的
            to_prune_idxs = torch.tensor(  # 创建空 tensor
                [],
                dtype=torch.long,
                device=context.device,
            )

        if not self._disable_prune and self.layer_idx < self.config.num_hidden_layers - 1:  # 如果没有禁用剪枝，且还不是最后一层
            context.update_aux_cache(to_prune_idxs, self.layer_idx)  # 将本层要剪掉的 token 保存到 Aux cache 中，供后续层使用

        if not self._disable_prune:  # 全局未禁用剪枝时
            context.prune(to_prune_idxs)  # 在 Context 内部真正删除这些 token（更新 hidden_states / KV / 各种索引）
        if is_prefill and not self._disable_prune and 'importance_scores_list' in locals():  # 如果 importance_scores_list 存在
            del importance_scores_list  # 显式删除以释放显存
        outputs = (context,)  # 按照上层约定，forward 返回一个元组，第一个元素是更新后的 Context
        return outputs  # 返回结果