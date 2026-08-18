import os
import pandas as pd 
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 1. Read stories from a local CSV file
csv_file_path = "Fired Storys.xlsx" 
df = pd.read_excel(csv_file_path)
# Target the stories column 'story_text'
column_name = 'Fired_Storys'
if column_name not in df.columns:
    column_name = df.select_dtypes(include=['object']).columns[0]

stories_list = df[column_name].astype(str).tolist()

# 2. Load Model & Tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3. Fast Dataset Class
class FastStoryDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts, 
            truncation=True, 
            max_length=max_length, 
            padding=False
        )

    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

dataset = FastStoryDataset(stories_list, tokenizer)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 4. Optimized Training Arguments
training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=50,
    per_device_train_batch_size=4, 
    gradient_accumulation_steps=2, 
    fp16=torch.cuda.is_available(), 
    dataloader_num_workers=0,
    save_steps=1000,
    logging_steps=20,
    save_total_limit=1
)

# 5. Train Model
trainer = Trainer(
    model=model, 
    args=training_args, 
    data_collator=data_collator, 
    train_dataset=dataset
)
trainer.train()

save_path = "C:\\Users\\LENOVO C\\Downloads\\VS Code\\Genrate Story\\Saves"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"Model trained and saved successfully to {save_path}!")