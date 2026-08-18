
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 1. Path to your fine-tuned model directory
model_path = "C:\\Users\\LENOVO C\\Downloads\\VS Code\\Genrate Story\\Saves"

print("Loading model and tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# 2. Enter the story prompt 
prompt = "my boss fired me"

# 3. Tokenize input prompt
inputs = tokenizer(prompt, return_tensors="pt")

print("Generating story...\n")

# 4. Generate story text
output = model.generate(
    **inputs, 
    max_length=300,           
    #num_return_sequences=1,  
    no_repeat_ngram_size=3,   
    do_sample=True,           
    top_k=50, 
    top_p=0.95,
    temperature=0.8
)

# 5. Decode output back into readable text
story = tokenizer.decode(output[0], skip_special_tokens=True)

print("=" * 40)
print("GENERATED STORY")
print("=" * 40)
print(story)