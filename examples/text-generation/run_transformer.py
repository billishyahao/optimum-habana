import torch
from transformers import pipeline
from transformers import GemmaForCausalLM
from transformers import Gemma2ForCausalLM
from transformers import LlamaForCausalLM
from optimum.habana.transformers.models.llama import GaudiLlamaModel
from optimum.habana.transformers.models.llama import GaudiLlamaModel
from optimum.habana.transformers.models.gemma import GaudiGemmaForCausalLM
from transformers.generation import GenerationConfig

pipe = pipeline(
    "text-generation",
    model="/intel/gemma-2-9b",
    device="cpu",  # replace with "mps" to run on a Mac device
)

text = "Once upon a time, there is"
outputs = pipe(text, max_new_tokens=64)
response = outputs[0]["generated_text"]
print(response)
