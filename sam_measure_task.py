import os
from langchain.llms import OpenAI
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent

os.environ["OPENAI_API_KEY"] = 'OPENAI_API_KEY'

llm = OpenAI(temperature=0)

template = '''\
You are an AI Assistant, your task is to extract the information from the input delimited by triple quote eg. sample_name, exposure_time, and angle.

A few samples are given below:
I want to measure (polymer)[sample_name] sample.
Collect data for (perovskite)[sample_name] material.
I want to measure a (polymer)[sample_name] sample.
Let's collect data for this (polymer)[sample_name] sample.
Exposure time of (8)[exposure_time] seconds.
Time (0.5)[exposure_time] seconds.
(two)[exposure_time] seconds.
I want to measure this (polymer)[sample_name] sample for (ten)[exposure_time] seconds at an incident angle of (0.1)[angle] degree
Incident angle of (0.5)[angle] degree.
Angle (0.3)[angle].
(0.1)[angle] degree.
Theta of (0.4)[angle].
Collect data for this material. Let us do an exposure time of (10)[exposure_time] seconds and incident angles at (0.1)[angle] and (0.2)[angle] degree.
Let's measure this (perovskite)[sample_name] sample.
We want to look at this (perovskite)[sample_name] to understand its structure. We think (5)[exposure_time] seconds of exposure should be sufficient. Theta of (0.2)[angle] would be good.

```
{query}
```
'''

prompt = PromptTemplate.from_template(template)

llm_chain = LLMChain(llm=llm, prompt=prompt)
result =llm_chain("Collect data for this material. Let us do an exposure time of 10 seconds and incident angles at 0.1 and 0.2 degree.")
print(result)

