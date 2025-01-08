from ollama import chat
from ollama import ChatResponse

def generate_enthymemes(input_file,output_file):
    with open(input_file, 'r') as source_file, open(output_file, 'w') as hypo_file:
        for line in source_file:
            # Strip whitespace and skip empty lines
            line = line.strip()
            if not line:
                continue

            # Generate the enthymeme
            try:
                response: ChatResponse = chat(model='llama3.2', messages=[
                    {
                        'role': 'user',
                        'content': line + ' Based on this statement, complete the argument as an enthymeme. Ensure the implicit premise provided after the # is included logically and clearly in the response.',
                    },
                ])

                # Extract the generated enthymeme
                enthymeme = response['message']['content']

                # Write the enthymeme to the output file
                hypo_file.write(enthymeme + '\n')
                print(f"Processed: {line}")
                print(f"Generated: {enthymeme}\n")

            except Exception as e:
                print(f"Error processing line: {line}\nError: {e}")
                hypo_file.write("ERROR\n")  # Log an error placeholder in the output file

generate_enthymemes('../Datasets/D1test/semevaldataparacomet.source','olama.hypo')



