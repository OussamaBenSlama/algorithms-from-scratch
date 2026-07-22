def tokenize_text(tokenizer, text):
    """Tokenize a single string and return a list[int] of input ids."""
    return tokenizer(text)["input_ids"]
