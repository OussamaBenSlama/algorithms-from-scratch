# pad_token : fills empty space so sequences have equal length
# pad_token_id : Integer ID representing the padding token
# eos_token : Marks the end of generated text
# attention_mask : Tells the model which tokens are real vs padding

def ensure_pad_token(tokenizer):
    """Guarantee tokenizer.pad_token is not None; fall back to eos_token."""
    if tokenizer.pad_token is None :
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
