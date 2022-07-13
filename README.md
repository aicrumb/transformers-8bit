# gpt-j-8bit
wrapper for hivemind's gpt-j-8bit training code for easy loading

check out my model finetuned with this style of loading: https://huggingface.co/crumb/gpt-j-6b-shakespeare

check out hivemind's original 8bit quantized model: https://huggingface.co/hivemind/gpt-j-6B-8bit


code:
```python
# libraries and a wrapper around hivemind's quantization code
!pip install transformers==4.14.1 bitsandbytes-cuda111==0.26.0 git+https://github.com/aicrumb/gpt-j-8bit -q
import gptj_8bit as gptj

# this should work with any gpt-j checkpoint, 8bit or not
model, tokenizer, config = gptj.load("crumb/gpt-j-6b-shakespeare", device='cuda')
gptj.generate(model, tokenizer, "Romeo:", min_length=64, max_length=64)

""" example output
Romeo: [Aside] And but in night, how tedious
Is the day's celebration!
JULIET: [Aside] O me! how quick skips time!
Bid Time himself look out And, after no long date,
Call time up o'er-head,
"""
```

finetuning code coming soon
