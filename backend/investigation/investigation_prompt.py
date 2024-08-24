
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

investigation_prompt = """
You are a world class therapist, who is helping your client have a consturctive discussion with their partner.
 - You must analyse the text provide and create 3 follow up questions to help them explore their feelings deeper.
 - I

{user_input}

 - You are part of a system so must return the questions in a json format with no other details

{format_instructions}
"""

response_schemas = [
        ResponseSchema(name="question_1", description="The first question"),
        ResponseSchema(name="question_2", description="The second question"),
        ResponseSchema(name="question_2", description="The third question")
    ]

    # Create the output parser

investigation_output_parser = StructuredOutputParser.from_response_schemas(response_schemas)