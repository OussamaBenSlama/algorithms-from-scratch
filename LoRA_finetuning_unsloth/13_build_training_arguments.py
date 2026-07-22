import torch 
from transformers import TrainingArguments

def build_training_arguments(output_dir='./sft_out', max_steps=5, learning_rate=2e-4):
    """Return featherweight TrainingArguments for the SFT run."""
    
    use_bf16 = torch.cuda.is_bf16_supported()
    return TrainingArguments (
        output_dir=output_dir,
        per_device_train_batch_size = 1 ,
        gradient_accumulation_steps = 1,
        max_steps = max_steps,
        learning_rate=learning_rate,
        bf16=use_bf16,
        fp16=not use_bf16,
        logging_steps=1,
        optim='adamw_8bit'
    )


"""
output_dir:Where checkpoints and logs are saved.

max_steps:Maximum number of optimizer steps.

learning_rate:Controls how big each weight update is.

per_device_train_batch_size=1:Each GPU processes one example at a time.

gradient_accumulation_steps=1:Update weights after every batch.

optim="adamw_8bit":Uses bitsandbytes 8-bit AdamW. Normal AdamW stores optimizer states in full precision


"""