import os
import nltk
import openai
from dotenv import load_dotenv
from typing import List, Union
from util import DELIMITER, GYM, FITNESS_ACTIVITY, TRAINING, JOURNEY


load_dotenv()


openai.organization = os.getenv("OPENAI_ORG_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")


SYSTEM = f"""
You are my assistant, you have extensive knowledge on physical fitness, \
you also have knowledge about my fitness routines including the days i go to the gym,\
fitness activity i do on each day, training routines on each activity and my journey to and from the gym each day. \
Take these steps to answer my queries which will be delimited by {DELIMITER}.

STATE 1: You need to classify which category my query belongs.\
The categories are GYM, FITNESS-ACTIVITY, TRAINING-ROUTINE, and JOURNEY. 
Your classification is allowed to be multi class if you think the query can be classed \
into more than one of the specified categories. Always remember that the classes \
include these four categories only, make sure not to classify \
any query outside of these four classes. 

STATE 2: Once my query has been classified, Follow the following conditions.

CONDITIONS

If classification is GYM, use {GYM} information to answer my queries 
If classification is FITNESS-ACTIVITY, use {FITNESS_ACTIVITY} information to answer my queries 
If classification is TRAINING-ROUTINE, use {TRAINING} information to answer my queries 
If classification is JOURNEY, use {JOURNEY} information to answer my queries 
If you think the query can not be classified into one of the specified categories,\
then combine all the information you have ({GYM}, {FITNESS_ACTIVITY}, {TRAINING}, {JOURNEY}) to give a \
responsible answer to my given query if applicable or reply by saying you do not have enough \
information to answer at this time.

Use the following format:
RESPONSE:{DELIMITER} <STATE 2 output>

IMPORTAN NOTES: 
Make sure your answers are responsible and follow the steps provided.

Always respond in second person
example
QUERY: what are my fitness activites?
RESPONSE: You do Muay thai, calisthenics, and weight training.
"""


def moderate(
        input: str
)-> str:
    """
    Send input text for moderation using the OpenAI API.

    Args:
        input (str): The text to be moderated.

    Returns:
        str: The moderation output from the API.

    """
    response = openai.Moderation.create(
    input=input
)
    moderation_output = response["results"][0]['flagged']
    return moderation_output


def assistant(
        context: list,
        debug: bool,
        model: str='gpt-3.5-turbo', 
        temperature: float=0,
) -> Union[str, tuple]:
    """
    Interact with the OpenAI Chat API to generate responses.

    Args:
        context (list): List of message dictionaries, each containing 'role' and 'content'.
        debug (bool): If True, returns token count and final response as a tuple.
        model (str, optional): The model to use for generating responses. Defaults to 'gpt-3.5-turbo'.
        temperature (float, optional): Temperature parameter for randomness in text generation. Defaults to 0.0.

    Returns:
        str or tuple: The generated response or (token_dict, final_response) tuple if debug is True.

    """
    messages = context
     
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature
        )
    result = response.choices[0].message["content"]

    # TOKEN COUNT
    token_dict = {
    'prompt_tokens':response['usage']['prompt_tokens'],
    'completion_tokens':response['usage']['completion_tokens'],
    'total_tokens':response['usage']['total_tokens'],
        }
    
    try:
        final_response = result.split(DELIMITER)[-1].strip()
    except Exception as e:
        final_response = "Sorry, I'm having trouble right now, please try asking another question."

    if debug: 
        return token_dict, final_response
    
    else:
        return final_response
    

context = [{"role": "system", "content":  SYSTEM}]


def respond(
        prompt: str,  
        debug: bool=False        
) -> str:
    """
    Respond to a user's prompt using the assistant and moderation checks.

    Args:
        prompt (str): The user's input prompt.
        debug (bool, optional): If True, additional information is printed for debugging. Defaults to False.

    Returns:
        str: The generated response or an error message.

    """
    user_moderation = moderate(prompt)
    if user_moderation:
        if debug:
            print(user_moderation)
            print("Input flagged by Moderation API.")
        return "Sorry, we cannot process this request."
    if debug: 
        print("Input passed moderation check.")

    global context
    context.append({"role": "user", "content": prompt})


    response = assistant(context, debug)
    if debug:
        assistant_moderation = moderate(response[1])
        if assistant_moderation:
            if debug:
                print("Step 5: Response flagged by Moderation API.")
            return "Sorry, we cannot provide this information."
        print("Response passed moderation check.")
        context.append({"role": "assistant", "content": response})
        return response[1]
    else:
        assistant_moderation = moderate(response[1])
        if assistant_moderation:
            return "Sorry, we cannot provide this information."
        context.append({"role": "assistant", "content": response})
        return response


if __name__ == "__main__":
    query = input()
    prompt = f"{DELIMITER}{query}"
    response = respond(prompt) 
    print(response)