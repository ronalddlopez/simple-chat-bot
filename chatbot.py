"""
Step 1: Installing requirements:

- pip3 install virtualenv 
- virtualenv my_env # create a virtual environment my_env
- source my_env/bin/activate # activate my_env

Step 2: Import our required tools from the transformers library
"""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)


"""
Step 3: Choosing a model
"""
model_name = "facebook/blenderbot-400M-distill"

"""
Step 4: Fetch the model and initialize a tokenizer

- # Load model (download on first run and reference local installation for consequent runs)
"""
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)

"""
Step 5.1: Keeping track of conversation history
"""
conversation_history = []

while True:
    """
    Step 5.2: Encoding the conversation history
    """
    history_string = "\n".join(conversation_history)

    """
    Step 5.3: Fetch prompt from user
    """
    input_text = input("> ")

    """
    Step 5.4: Tokenization of user prompt and chat history
    """
    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")
    # print(input_text)

    # tokenizer.pretrained_vocab_files_map

    """
    Step 5.5: Generate output from the model
    """
    outputs = model.generate(**inputs, max_new_tokens=120)

    #print(outputs)

    """
    Step 5.6: Decode output
    """
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    print(response)

    conversation_history.append(input_text)
    conversation_history.append(response)
    # print(conversation_history)