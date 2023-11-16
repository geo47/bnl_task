import os

from langchain.chains import LLMChain
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from torch import cuda, bfloat16
from transformers import AutoTokenizer
import transformers

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

model_name = 'meta-llama/Llama-2-7b-chat-hf'

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name,
    # trust_remote_code=True,
    # config=transformers.AutoConfig.from_pretrained(model_name),
    # quantization_config=transformers.BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type='nf4',
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=bfloat16),
    # device_map='cuda:0',
)
# enable model inference
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)
pipeline = transformers.pipeline(
    task='text-generation',
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,  # needed by langchain
    # model params
    max_new_tokens=512,
    temperature=0.1,  # creativity of responses: 0.0 none ->  1.0 max
    repetition_penalty=1.1  # to avoid repeating output
)

hf_llm = HuggingFacePipeline(pipeline=pipeline)

# chat_prompt = PromptTemplate.from_template("""
# [INST] <<SYS>>You are a helpful assistant that will extract information from user input. Your task is to extract sample_name, exposure_time, and angle from user input.
# sample_name is optional while exposure_time and angle are the required entities. If user input doesn't contain exposure_time or angle entities, then ask user to provide values for the mising entities.
#
# Below are a few annotated examples of the entities that you need to extract:
# examples: |
#     - I want to measure this sample
#     - Collect data for this material
#     - We want to look at this sample to understand its structure
#     - Can we measure this sample
#     - Can we measure this material
#     - I want to measure a [polymer sample](sample_name)
#     - Let's collect data for this [polymer sample](sample_name)
#     - Exposure time of [8](exposure_time) seconds
#     - Time [0.5](exposure_time) seconds
#     - [two](exposure_time) seconds
#     - I want to measure this [polymer sample](sample_name) for [ten](exposure_time) seconds at an incident angle of [0.1](angle) degree
#     - Incident angle of [0.5](angle) degree
#     - Angle [0.3](angle)
#     - [0.1](angle) degree
#     - Theta of [0.4](angle)
#     - Collect data for this material. Let us do an exposure time of [10](exposure_time) seconds and incident angles at [0.1](angle) and [0.2](angle) degree
#     - Let's measure this [perovskite sample](sample_name)
#     - We want to look at this [perovskite sample](sample_name) to understand its structure. We think [5](exposure_time) seconds of exposure should be sufficient. Theta of [0.2](angle) would be good.
# <</SYS>>
# {content}[/INST]""")

chat_prompt = PromptTemplate.from_template("""
[INST] <<SYS>>You are a helpful Mathematician
<</SYS>>
{content}[/INST]""")

# query = "user input: I want to measure this polymer sample for ten seconds"
query = """
Given that
d = 2π / q
q = 4π sin(θ) / λ

If q is 1, and the wavelength lambda λ is 1, what is the scattering angle θ in degree? What is the d spacing?
"""
# content = "user input: {content}"
print(chat_prompt.format(content=query))

chain = LLMChain(llm=hf_llm, prompt=chat_prompt)


response = chain.run(content=chat_prompt.format(content=query))
response = response.replace("\n", "")

print(response)