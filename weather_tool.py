#
# weather_tool.py
# This module contains all ingredients to build a langchain tool
# that incapsule any custom function.
#
import json
from langchain.agents import Tool


#
# weather_data
# is an example of a custom python function
# that takes a list of custom arguments and returns a text (or in general any data structure)
#
def weather_data(where: str = None, when: str = None) -> str:
    '''
    given a location and a time period, this custom function
    returns weather forecast description in natural language.

    This is a mockup function, returning a fixed text tempalte.
    The function could wrap an external API returning realtime weather forecast.

    parameters:
        where: location as text, e.g. 'Genova, Italy'
        when: time period, e.g. 'today, now'

    returns:
        weather foreast description as flat text.
    '''
    if where and when:
        # return a fake/hardcoded weather forecast sentence
        return f'in {where}, {when} is sunny! Temperature is 20 degrees Celsius.'
    elif not where:
        return 'where?'
    elif not when:
        return 'when?'
    else:
        return 'I don\'t know'


def weather(json_request: str) -> str:
    '''
    Takes a JSON dictionary as input in the form:
        { "when":"<time>", "where":"<location>" }

    Example:
        { "when":"today", "where":"Genova, Italy" }

    Args:
        request (str): The JSON dictionary input string.

    Returns:
        The weather data for the specified location and time.
    '''
    arguments_dictionary = json.loads(json_request)
    where = arguments_dictionary["where"]
    when = arguments_dictionary["when"]
    result = weather_data(where=where, when=when)
    return result


#
# instantiate the langchain tool.
# The tool description instructs the LLM to pass data using a JSON.
# Note the "{{" and "}}": this double quotation is needed to avoid a runt-time error triggered by the agent instatiation.
#
name = "weather"
request_format = '{{"when":"<time>","where":"<location>"}}'
description = f'helps to retrieve weather forecast. Input should be JSON in the following format: {request_format}'

# create an instance of the custom langchain tool
Weather = Tool(name=name, func=weather, description=description)


if __name__ == '__main__':
    print(weather_data(where='Genova, Italy', when='today'))
    # => in Genova, Italy, today is sunny! Temperature is 20 degrees Celsius.

    print(weather('{ "when":"today", "where":"Genova, Italy" }'))
    # => in Genova, Italy, today is sunny! Temperature is 20 degrees Celsius.

    # print the Weather tool
    print(Weather)