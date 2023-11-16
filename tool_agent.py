#
# tools_agent.py
#
# zero-shot react agent that reply questions using available tools
# - Weater
# - Datetime
#
# get the question as a command line argument (a quoted sentence).
# $ py tools_agent.py What about the weather today in Genova, Italy
#
import os
import sys

from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain import LLMChain
from langchain.prompts import PromptTemplate

# import custom tools
from weather_tool import Weather
from datetime_tool import Datetime

os.environ["OPENAI_API_KEY"] = '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'

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

prompt = PromptTemplate(input_variables=['query'], template=template)

# debug
# print(prompt.format())
# sys.exit()


# Load the tool configs that are needed.
llm_weather_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True
)

tools = [
    Weather,
    Datetime
]

# Construct the react agent type.
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)

if __name__ == '__main__':
    question = "what the weather in genova?"
    print('question: ' + question)

    # run the agent
    agent.run(question)
