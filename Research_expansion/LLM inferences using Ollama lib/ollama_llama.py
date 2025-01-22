from ollama import chat
from ollama import ChatResponse

#zero shot with llama3.2
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
                        'content': 'The line first has a sentence followed by one or more implicit premises. Create a concise sentence by incorporating the implicit premises with the sentence. Only provide me with the result that is one sentence without any additional information. Here is the line: ' + line,
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
                #hypo_file.write("ERROR\n")  # Log an error placeholder in the output file

#few shot with llama3.2
def generate_enthymemes_with_examples(input_file,output_file):
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
                        'content': 'The line first has a sentence followed by one or more implicit premises. Create a concise sentence by incorporating the implicit premises with the sentence. Only provide me with the result without any additional information. Here are some examples. source: Interns are replacing employees. # to have a better job # Unpaid internship exploit college students. target: Interns are replacing employees. And since that helps the companys bottom line. Unpaid internship exploit college students. source: If home schooled children get to pick a high school sports team high school kids will want to be able to play for other schools. # to have fun # Home-schoolers should not play for high school teams. target: If home schooled children get to pick a high school sports team high school kids will want to be able to play for other schools. And since picking teams is not the way the system works. Home-schoolers should not play for high school teams. source: Labeling a product with "natural" is an attempt to portray the food as "healthy", but is often not the case. # to be healthy # Restrict use of the word "natural" on food labels. target: Labeling a product with "natural" is an attempt to portray the food as "healthy", but is often not the case. And since most people dont understand that natural does not mean healthy. Restrict use of the word "natural" on food labels. Here is the line: ' + line,
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

#generate_enthymemes('../Datasets/D1test/semevaldataparacomet.source', 'semevaldata.hypo')
generate_enthymemes_with_examples('../Datasets/D1test/semevaldataparacomet.source', 'semevaldata_fewshot.hypo')



