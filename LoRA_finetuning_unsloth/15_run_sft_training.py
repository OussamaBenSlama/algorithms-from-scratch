def run_sft_training(trainer):
    """Run a few SFT steps and return the final training loss as a float."""
    
    result = trainer.train()

    final_loss = result.metrics["train_loss"]
    return float(final_loss)