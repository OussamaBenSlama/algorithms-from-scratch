def get_lora_target_modules():
    """Return the attention projection module name suffixes for LoRA."""
    return ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    # query, key, value, and output projections (Full attention LoRA)

"""
A target module tells LoRA Which layers should receive these small trainable LoRA matrices
A transformer model has many layers:
Qwen2.5
│
├── Embedding
│
├── Transformer Block 1
│   ├── q_proj
│   ├── k_proj
│   ├── v_proj
│   ├── o_proj
│   ├── gate_proj
│   ├── up_proj
│   └── down_proj
│
├── Transformer Block 2
│   └── ...

"""