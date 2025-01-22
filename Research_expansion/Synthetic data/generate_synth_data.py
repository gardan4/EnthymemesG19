from ollama import chat
from ollama import ChatResponse

MODELS = ['phi4', 'qwen2.5:14b']
NUM_ARGS_TO_GENERATE = 10000
PROMPT = "Give me a short argument, no longer than 3 sentences. Then give me the same sentence but for one crucial difference: a part of the argument has been removed, specifically one that tends to be understood by people implicitly. For example, provide me with a sentence like 'The concept of standardized testing aims to provide a uniform measure across different schools and regions to evaluate student performance on common standards. And since these tests are just one aspect of educational assessment, it's logical that they contribute only partially to overall academic grades (often around 20% weighting).' Then, also provide me another sentence like: 'The concept of standardized testing aims to provide a uniform measure across different schools and regions to evaluate student performance on common standards. # It's logical that they contribute only partially to overall academic grades (often around 20% weighting).' Notice how they are identical, except for the fact part of the first sentence has been replaced by a '#'. This part is not crucial to state explicitly for the reader to make sense of the argument. I want your response to be in the following format: First, the original argument. Then the same argument except a part has been replaced by a '#' as shown in my example. The argument should concern a random topic. ADD NO OTHER TEXT TO YOUR RESPONSE!"
PATH_TO_TARGET_FILE = 'train.target'
PATH_TO_SOURCE_FILE = 'train.source'

def get_prompt_response(prompt, model):
    response: ChatResponse = chat(model=model, messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])

    return response['message']['content']

def check_if_valid_response_and_format(response):
    # Separate the lines of the response
    separated_response = response.replace('’', "'").split('\n')

    # Check that there are least two lines in the response: one for the target string and one for the source string
    if len(separated_response) < 2:
        return None
    
    # Assume that the first line corresponds to the target string, and the last one to the source string (ignoring potential separating lines between them)
    target = separated_response[0].rstrip()
    source = separated_response[len(separated_response)-1].rstrip()

    # Check that the target string has no hashtags, and the source string has exactly one hashtag
    if target.count('#') != 0 or source.count('#') != 1:
        return None
    
    # Separate the target string before and after the hashtag
    source_separated = source.split('#')

    # Check that there are exactly two elements of the split string: the part before the hashtag and the part after
    if len(source_separated) != 2:
        return None
    
    # Check that both parts are are least 30 characters long
    if len(source_separated[0]) < 30 or len(source_separated[1]) < 30:
        return None
    
    # Check that the first and last parts of the target and source strings are equivalent
    if source_separated[0][:-1].lower() != target[:len(source_separated[0])-1].lower() \
    or source_separated[1][1:].lower() != target[-len(source_separated[1])+1:].lower():
        return None

    # Check if a reasonable portion of the target string has been removed from the source string
    if len(target) < len(source) + 30:
        return None
    
    return (target, source)

def add_line_to_file(file_path, string):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    lines.append(string + '\n')

    with open(file_path, 'w') as file:
        file.writelines(lines)

args_remaining = NUM_ARGS_TO_GENERATE
current_model = MODELS[0]

while args_remaining:
    response = get_prompt_response(PROMPT, current_model)
    formatted_response = check_if_valid_response_and_format(response)

    # If the response message was not in the expected format, try again
    if not formatted_response:
        print("Response format invalid; will try again.")
        continue

    target = formatted_response[0]
    source = formatted_response[1]

    add_line_to_file(PATH_TO_TARGET_FILE, target)
    add_line_to_file(PATH_TO_SOURCE_FILE, source)

    args_remaining -= 1

    print(f"Added argument created using {current_model}. {args_remaining} arguments remaining.")

    if current_model == MODELS[0]:
        current_model = MODELS[1]
    else:
        current_model = MODELS[0]