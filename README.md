# transformers-8bit
wrapper for hivemind's gpt-j-8bit training code for easy loading

model card of a model that i trained with this as proof of concept: https://huggingface.co/crumb/gpt-j-6b-shakespeare

check out hivemind's original 8bit quantized model: https://huggingface.co/hivemind/gpt-j-6B-8bit

### inference example

```python
# libraries and a wrapper around hivemind's quantization code
!pip install transformers==4.14.1 bitsandbytes-cuda111==0.26.0 git+https://github.com/aicrumb/transformers-8bit -q
import transformers_8bit

# this should work with any gpt-j checkpoint, 8bit or not
model, tokenizer, config = transformers_8bit.load_gptj("crumb/gpt-j-6b-shakespeare", device='cuda')

prompt = tokenizer("Romeo:", return_tensors='pt')
prompt = {key: value.to('cuda') for key, value in prompt.items()}
out = model.generate(**prompt, min_length=64, max_length=64, do_sample=True, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(out[0]))

""" example output
Romeo: [Aside] And but in night, how tedious
Is the day's celebration!
JULIET: [Aside] O me! how quick skips time!
Bid Time himself look out And, after no long date,
Call time up o'er-head,
"""
```

### finetuning example

```python
from IPython import display 
!pip install transformers==4.14.1 datasets -q
!pip install bitsandbytes-cuda111==0.26.0 -q
!pip install git+https://github.com/aicrumb/datasettokenizer -q
!pip install git+https://github.com/aicrumb/transformers-8bit -q

import transformers
import torch
from bitsandbytes.optim import Adam8bit
from tqdm.auto import tqdm
import transformers_8bit
import datasettokenizer as tok

from huggingface_hub import notebook_login
notebook_login() # log in before running rest of cells

# load model
model, tokenizer, config = transformers_8bit.gptj(device='cuda')

# load and tokenize dataset
from datasets import load_dataset
dataset = load_dataset("tiny_shakespeare")
# dataset = load_dataset("text", data_files={"train": "train.txt", "validation": "test.txt"})
dataset = tok.tokenize_dataset(dataset, tokenizer, block_size=256)

# train model
checkpoint_name = "gpt-j-6b-finetune"
training_args = transformers.TrainingArguments(
    checkpoint_name,
    push_to_hub=True,
    num_train_epochs=1,
    save_strategy="epoch",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8
)

trainer = transformers.Trainer(
    model=model,
    optimizers=(
        Adam8bit(model.parameters(), lr=3e-4, weight_decay=0.01),
        None
    ),
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)
trainer.train()

# push model to huggingface hub
model.push_to_hub(checkpoint_name)
```

