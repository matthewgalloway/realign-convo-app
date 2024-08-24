from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from investigation.investigation_prompt import investigation_prompt, investigation_output_parser
from advice.advice_prompt import advice_prompt, advice_output_parser

import os
import logging

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:8080"}})

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = app.logger

openai_api_key = os.environ.get('OPENAI_API_KEY')
if not openai_api_key:
    logger.error("No OpenAI API key found in environment variables")
    raise ValueError("No OpenAI API key found in environment variables")

@app.route('/api/generate-questions', methods=['POST'])
def generate_questions():
    print(f"Request method: {request.method}")
    print(f"Request headers: {request.headers}")
    print(f"Request data: {request.data}")
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()
    try:
        data = request.json
        if not data:
            logger.warning("No JSON data received")
            return jsonify({"error": "No JSON data received"}), 400
        
        user_input = data.get('text')
        if not user_input:
            logger.warning("No 'text' field in JSON data")
            return jsonify({"error": "No 'text' field in JSON data"}), 400
        
        logger.info(f"Generating questions for input: {user_input}")
        
        # Initialize ChatOpenAI model
        model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, api_key=openai_api_key)

        # Create the prompt template
        prompt = ChatPromptTemplate.from_template(investigation_prompt)

        # Create the runnable sequence
        chain = (
            {"user_input": RunnablePassthrough(), "format_instructions": lambda _: investigation_output_parser.get_format_instructions()}
            | prompt
            | model
            | StrOutputParser()
        )
        # Run the chain
        result = chain.invoke(user_input)
        
        # Parse the output
        parsed_output = investigation_output_parser.parse(result)
        logger.info(f"Generated questions: {parsed_output}")
        return jsonify({"questions": parsed_output})

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    


@app.route('/api/generate-speech', methods=['POST'])
def generate_speech():
    try:
        data = request.json
        if not data:
            logger.warning("No JSON data received")
            return jsonify({"error": "No JSON data received"}), 400
        
        logger.info(f"Received data for speech generation: {data}")
        
        original_text = data.get('originalText')
        questions = data.get('questions')
        answers = data.get('answers')
       
        if not original_text or not questions or not answers:
            logger.warning("Missing required fields")
            return jsonify({"error": "Missing required fields"}), 400
        
        question_answers = "\n".join([f"Question: {q}\nAnswer: {a}" for q, a in zip(questions, answers)])
        
        # Initialize ChatOpenAI model
        model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, api_key=openai_api_key)

        # Create the prompt template
        prompt = ChatPromptTemplate.from_template(advice_prompt)

        # Create the runnable sequence
        chain = (
            {"original_text": RunnablePassthrough(),
             "question_answers": RunnablePassthrough(),
              "format_instructions": lambda _: advice_output_parser.get_format_instructions()}
            | prompt
            | model
            | StrOutputParser()
        )
        # Run the chain
        result = chain.invoke({
                    "original_text": original_text,
                    "question_answers": question_answers
                })

        # Parse the output
        processed_speech = advice_output_parser.parse(result)
        logger.info(f"Generated speech: {processed_speech}")
        return jsonify({"speech": processed_speech})

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')