from dataclasses import dataclass

@dataclass
class SmolLM2Config:
    vocab_size: int = 49152
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    initializer_range: float = 0.041666666666666664
    
    # Training specific parameters
    learning_rate: float = 3e-4
    batch_size: int = 8
    context_length: int = 256  # Smaller context length for training
    num_epochs: int = 10 