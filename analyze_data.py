from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_community.chat_models import ChatOCIGenAI

from config import ANALYZER_MODEL_ID, SERVICE_ENDPOINT, TEMPERATURE, MAX_TOKENS
from config_private import COMPARTMENT_ID

SYSTEM_PROMPT = """
You are a data analysis tool. You are given a prompt and a dataset.
You need to analyze the data based on the prompt and return the result.
"""

PROMPT_TEMPLATE = """
Based on the following data, answer the user's prompt.

Data: {DATA}

Prompt: {PROMPT}
"""


@tool
def data_analyzer(original_prompt: str, data: str):
    """Provides insights, trends, or analysis based on the data and prompt.

    Args:
        original_prompt (str): The original user prompt that the data is based on.
        data (str): The data to analyze.

    Returns:
        str: The analysis result.
    """
    print("Called data_analyzer...")

    model = ChatOCIGenAI(
        auth_type="API_KEY",
        model_id=ANALYZER_MODEL_ID,
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
        is_stream=True,
        model_kwargs={"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS},
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=PROMPT_TEMPLATE.format(PROMPT=original_prompt, DATA=data)),
    ]

    response = model.invoke(messages)

    return response.content
