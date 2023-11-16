import os

import pandas as pd


# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import transformers
from langchain.chains import LLMChain
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer

from TaskRecognition.rec_model import ExtractNER
from task1_main import task1_bot

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


if __name__ == '__main__':
    ex_feature = ExtractNER()

    # task_dataset = pd.read_csv("TaskRecognition/dataset/raw_files/task_dataset.csv", header=0)

    # for idx, row in task_dataset.iterrows():
    #     ex_feature.predict_ner(row[0])

    # query = "If q is 1, and the wavelength lambda λ is 1, what is the scattering angle θ in degree? What is the d spacing?"
    # query = "If a material exhibits a spacing between crystal lattice planes d of 6, what does this correspond to in terms of the reciprocal/Fourier space scattering vector q?"
    query = "We want to look at this perovskite to understand its structure. We think 5 seconds of exposure should be sufficient. Theta of 0.2 would be good."

    result = ex_feature.predict_ner(query)
    print(result['ner'])

    task_no = ""
    for ner_obj in result['ner']:
        if 'task_no' in ner_obj:
            task_no = ner_obj["task_no"]
            break

    if task_no:
        if task_no == "task_1":
            task1_bot(ex_feature, result['ner'])
            pass
        if task_no == "task_6":


            model_name = 'meta-math/MetaMath-7B-V1.0'
            # model_name = 'meta-llama/Llama-2-7b-chat-hf'

            model = transformers.AutoModelForCausalLM.from_pretrained(model_name)

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

            math_prompt = PromptTemplate.from_template("""
[INST] <<SYS>> You are a helpful Mathematician. Given two equations, your task is to analyze the problem and select appropriate equation(s) to solve the problem. 

Given that
d = 2π / q
q = 4π sin(θ) / λ

<</SYS>>
{content}[/INST]""")

            # math_prompt = PromptTemplate.from_template("""
            # [INST] <<SYS>> You are a helpful Mathematician. Given two equations, your task is to analyze the problem and use appropriate equation(s) to solve the problem.
            #
            # Given that
            # equation 1: d = 2π / q
            # equation 2: q = 4π sin(θ) / λ
            #
            # equation 1 is used to calculate the spacing between crystal lattice planes
            # equation 2 is the scattering vector formula.
            #
            # where:
            # - q is the scattering vector,
            # - π is a mathematical constant (approximately 3.14159),
            # - θ is the scattering angle, and
            # - λ is the wavelength of the incident radiation.
            # <</SYS>>
            # {content}[/INST]""")



            math_query = query

            print(math_query)

            print(math_prompt.format(content=math_query))

            chain = LLMChain(llm=hf_llm, prompt=math_prompt)

            response = chain.run(content=math_prompt.format(content=math_query))
            response = response.replace("\n", "")

            print(response)
