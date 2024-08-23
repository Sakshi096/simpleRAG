
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
generation_model = GPT2LMHeadModel.from_pretrained('gpt2')

# Function to generate text based on the retrieved documents
def generate(retrieved_docs):
    input_text = " ".join(retrieved_docs)
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    outputs = generation_model.generate(inputs, max_length=100, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
