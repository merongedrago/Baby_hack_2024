import google.generativeai as genai
import os
from dotenv import load_dotenv


load_dotenv()
# Directly set your API key
api_key = os.getenv("API_KEY")
# Replace with your actual API key

# Configure the API key
genai.configure(api_key=api_key)

gemini_client = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction="You are a helpful assistant and an expert on child safety",
)


def get_completion_gemini(prompt, model="gemini-1.5-flash"):
    messages = [
        {
            "role": "model",
            "parts": ["You are a helpful assistant and an expert on child safety"],
        },
        {"role": "user", "parts": [prompt]},
    ]

    response = gemini_client.generate_content(
        messages,
        generation_config=genai.GenerationConfig(
            temperature=0,
        ),
    )

    return response.text.strip()  # Clean response


# Safe items to be around a baby
safe_items_for_baby = []

# Dangerous items for a baby
dangerous_items_for_baby = []


def check_dangerous_items(main_list, result_dic):

    # Convert lists to sets for O(1) lookups

    for item in main_list:
        if item in result_dic:
            pass
        else:
            # Check with Gemini for unknown items

            prompt = f"Is '{item}' dangerous for a baby? Give me a 0 if it is not dangerous and 1 if it is dangerous. Do not include any other text other than a 0 or 1."
            response = get_completion_gemini(prompt)

            cleaned_response = response.strip()  # Clean response
            danger_value = int(cleaned_response)  # Convert to int

            result_dic[item] = danger_value  # Store the value in the result dictionary

    return result_dic
