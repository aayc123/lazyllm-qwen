from transformers import Qwen3Config

class LazyQwen3Config(Qwen3Config):
    """
    A configuration class that extends the Qwen3Config with support for pruning rates. 
    Used for both LazyLlamaModel and LazyLlamaForCausalLM.
    """
    def __init__(
            self,
            pruning_rates: dict, 
            **kwargs
        ):
        """
        Initializes the LazyLlamaConfig with pruning rates and other LlamaConfig parameters.

        Args:
            pruning_rates (dict): A dictionary specifying the pruning rate for each layer. The format is {layer_index: pruning_rate} 
                starting at 0. The pruning rate is a float value between 0 and 1, where 0 means all tokens are preserved and 1 
                means only the last token is preserved.
            **kwargs: Other arguments passed to the base LlamaConfig class.

        """
        self.pruning_rates = pruning_rates
        super().__init__(**kwargs)

    @staticmethod
    def from_qwen3_config(pruning_rates, config: Qwen3Config):
        """从已有的 Qwen3Config 完整构造 LazyQwen3Config。

        之前这里是手工挑字段拷贝，只保留了部分关键参数，
        像 rope_theta、bos/eos/pad_token_id、use_cache 等很多配置都丢失，
        会导致 Lazy 模型和原始 Qwen3 的行为不一致，从而严重影响生成质量。

        现在直接基于 config.to_dict() 复制全部配置字段，仅额外加入 pruning_rates。
        """
        cfg_dict = config.to_dict()
        # 避免重复字段冲突
        cfg_dict.pop("pruning_rates", None)
        return LazyQwen3Config(pruning_rates=pruning_rates, **cfg_dict)