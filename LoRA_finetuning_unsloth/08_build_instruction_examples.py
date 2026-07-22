def build_instruction_examples():
    """Return a small list of {'instruction', 'response'} dicts for SFT."""
    return [
        {
            "instruction": "What is the capital of France?",
            "response": "The capital of France is Paris."
        },
        {
            "instruction": "Translate 'hello' into Spanish.",
            "response": "The Spanish translation of 'hello' is 'hola'."
        },
        {
            "instruction": "Write a short greeting.",
            "response": "Hello! I hope you're having a great day."
        },
        {
            "instruction": "What is 5 + 7?",
            "response": "5 + 7 = 12."
        }
    ]
