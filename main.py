import argparse
import re
import time
import logging
from ast import literal_eval

import litellm
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

def main(args):
    with open(args.prompt) as f:
        prompt = f.read()

    messages = [{'role': 'system', 'content': prompt}]
    log_file = f"experiments/{args.name}.txt"

    while True:
        # 1. Accept user input.
        user_input = input("[ PATIENT ]:\n")
        _save(log_file, f"[ PATIENT ]:\n{user_input}")

        if user_input == "END":
            print("[AI DOCTOR]:\nGoodbye!")
            _save(log_file, "[AI DOCTOR]:\nGoodbye!")
            break

        # 2. Generate response.
        messages.append({'role': 'user', 'content': user_input})
        response = litellm.completion(
            model=f"{args.provider}/{args.model}", messages=messages,
            temperature=args.temperature,
        ).choices[0].message.content

        # 3. Parse response.
        parsed_response = _parse_json(response)
        if not parsed_response:
            break

        # 4. Save response.
        print(f"[ THOUGHT ]:\n{parsed_response['thought']}\n[ AI DOCTOR ]:\n{parsed_response['speech']}")
        _save(log_file, f"[ THOUGHT ]:\n{parsed_response['thought']}")
        _save(log_file, f"[ AI DOCTOR ]:\n{parsed_response['speech']}")
        messages.append({'role': 'assistant', 'content': str(parsed_response)})

    print("~~ END OF CONVERSATION ~~")
    _save(log_file, "~~ END OF CONVERSATION ~~")

def _parse_json(text:str):
    matches = re.findall(r'\{.*?\}', text, re.DOTALL)
    if not matches:
        logger.error("Failed to extract JSON: %s", text)
        return None
    try:
        return literal_eval(matches[0])
    except Exception as e:
        logger.error("Failed to parse JSON. Fixing...")
        fixed_json = _fix_json(matches[0], str(e))
        return _parse_json(fixed_json)

def _fix_json(text:str, error:str):
    prompt = (
        "The following JSON is invalid when being parsed by Python's `ast.literal_eval` function. "
        f"JSON:\n{text}\nError:\n{error}\n"
        "Output the fixed JSON below."
    )
    response = litellm.completion(
        model="openai/gpt-4", messages=[{'role': 'user', 'content': prompt}],
        temperature=0.01
    ).choices[0].message.content
    return response

def _save(file_name:str, content: str):
    with open(file_name, "a") as f:
        f.write(content + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Physician")
    parser.add_argument("--name", type=str, required=True, help="Name of the experiment.")
    parser.add_argument("--prompt", type=str, default="prompt.txt", help="System prompt.")
    parser.add_argument("--model", type=str, default="gpt-4", help="Model name to use for inference.")
    parser.add_argument("--provider", type=str, default="openai", help="Model provider.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for sampling.")
    args = parser.parse_args()

    main(args)
