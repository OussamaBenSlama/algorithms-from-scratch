from unsloth import FastLanguageModel


def attach_lora_adapters(model, r=8, lora_alpha=16, target_modules=None):
    """Wrap the base model with LoRA adapters and return the PEFT model."""
    # TODO: wrap `model` with LoRA via FastLanguageModel.get_peft_model using r, lora_alpha, target_modules
    model = FastLanguageModel.get_peft_model(
        model=model,
        r=r,
        target_modules = target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
    )


"""
r : The LoRA rank.
means LoRA creates small matrices with rank 8:

Original weight:
W
LoRA update:
B(4096 x 8) @ A(8 x 4096)

lora_alpha:Scaling factor
Controls the strength of the LoRA update.

target_modules: Where LoRA should be inserted.
"""