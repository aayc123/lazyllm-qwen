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

    def from_qwen3_config(pruning_rates, config: Qwen3Config):
        return LazyQwen3Config(
            pruning_rates=pruning_rates,
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,           
            hidden_act=config.hidden_act,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            rope_scaling=config.rope_scaling,
            attention_bias=config.attention_bias,
            attention_dropout=config.attention_dropout,
            use_sliding_window=config.use_sliding_window,  
            sliding_window=config.sliding_window,          
            max_window_layers=config.max_window_layers,   
            layer_types=config.layer_types,                
        )