from transformers import pipeline
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Check if API key is loaded correctly
if not OPENAI_API_KEY:
    raise ValueError("API key not found. Please check your .env file.")
else:
    print("API key loaded successfully")

client = OpenAI()
chatbot = pipeline("text-generation", model="mistralai/Mistral-7B-v0.1")

while True:
    user_input = input("Enter your question: ")
    if user_input.lower() == "exit":
        print("Exiting the chatbot. Goodbye!")
        break
    else:
        print("Using OpenAI model...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            max_tokens=50,  # The maximum number of tokens to generate in output
            n=1,  # The number of responses to generate
            temperature=0.7,  # Controls randomness
            messages=[
                {"role": "system", "content": user_input},
            ]
        )

        for choice in response.choices:
            print(f"OpenAI response: {choice.message.content}")

        print("Using Mistral AI model...")
        mistral_response = chatbot(user_input, max_length=50)
        print(f"Mistral response: {mistral_response[0]['generated_text']}")