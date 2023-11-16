import os

from langchain.chains import LLMChain
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


def get_response_from_query(query):
    """
    Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    # docs = db.similarity_search(query, k=k)
    # docs_page_content = " ".join([d.page_content for d in docs])

    model_id = "google/flan-t5-xl"
    # chat = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 1e-10})

    config = AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, config=config)

    _pipeline = pipeline('text2text-generation',
                         model=model,
                         tokenizer=tokenizer,
                         max_length=512
                         )

    hf_llm = HuggingFacePipeline(pipeline=_pipeline)

    # Template to use for the system message prompt
    template = """
        You are a helpful assistant that that will call the sam_measure(sample_name:string, exposure_time=float, angle=list[float]) function after getting all the required parameters from the user input.

        sample_name is optional while exposure_time and angle are the required parameters.
        
        Only ask the missing parameters required to run the sam_measure(sample_name:string, exposure_time=float, angle=list[float]) function from the user input.

        Once you get all the required parameters, execute the following function:
        
        def sam_measure(sample_name:string, exposure_time=float, angle=float):
            print("Executing hardware with params: sample_name, exposure_time, angle"
        
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=hf_llm, prompt=chat_prompt)

    response = chain.run(question=query)
    response = response.replace("\n", "")
    return response

"""
   Inputting a query and calling the LLM
"""
query = "I want to measure this polymer sample for ten seconds"
output = get_response_from_query(query)
print(output)
