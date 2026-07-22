def count_trainable_parameters(model):
    """Return the number of trainable parameters in `model`."""
    count = 0
    for param in model.parameters():
        if param.requires_grad == True: # gradients enabled
            count += param.numel()
    return count

