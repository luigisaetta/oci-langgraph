from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

FAKE_DATA = """Company,Total amount
               Oracle, 1000
               IBM, 100
               MS, 500
            """


@tool
def generate_and_run_sql_query(original_prompt: str):
    """Generates and runs an SQL query based on the prompt.

    Args:
        original_prompt (str): A string containing the original user prompt.

    Returns:
        str: The result of the SQL query.
    """
    print("Called generate_and_run_sql_query...")

    return FAKE_DATA
