def generate_reply(model, tokenizer, prompt, max_new_tokens=32):
    """Greedy-generate a reply for `prompt` and return the decoded text."""
    
    inputs = tokenizer(prompt,return_tensors="pt")

    inputs = {
        key : value.to(model.device) for key,value in  inputs.items()
    }

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

    return response

