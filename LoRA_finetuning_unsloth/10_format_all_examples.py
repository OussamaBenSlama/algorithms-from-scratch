def format_all_examples(examples):
    """Format each instruction/response dict into a training string."""
    # TODO: apply format_instruction_example to every example and return the list
    training = []
    for example in examples:
        training.append(format_instruction_example(example))
    return training
