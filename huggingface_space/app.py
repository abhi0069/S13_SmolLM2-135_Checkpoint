import gradio as gr
import torch
from model import SmolLM2
from config import SmolLM2Config
from tokenizers import Tokenizer

def load_model():
    config = SmolLM2Config()
    model = SmolLM2(config)
    # Load your trained model
    model.load_state_dict(torch.load("checkpoints/final_5050.ckpt", map_location='cpu'))
    model.eval()
    return model

def generate_text(prompt, max_length=100, temperature=0.7):
    # Tokenize input
    input_ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            # Apply temperature
            next_token_logits = outputs[0, -1, :] / temperature
            # Sample from the distribution
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
    return tokenizer.decode(input_ids[0].tolist())

# Initialize model and tokenizer
model = load_model()
tokenizer = Tokenizer.from_file("tokenizer.json")

# Create Gradio interface
demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=2, label="Enter your prompt", placeholder="First Citizen:"),
        gr.Slider(minimum=10, maximum=200, value=100, step=1, label="Maximum Length"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature")
    ],
    outputs=gr.Textbox(lines=5, label="Generated Text"),
    title="SmolLM2-135M Shakespeare Text Generator",
    description="""This is an implementation of SmolLM2-135M, trained on Shakespeare's text.
                   Enter a prompt and adjust the generation parameters to create Shakespeare-style text.""",
    examples=[
        ["First Citizen:", 100, 0.7],
        ["MARCIUS:", 100, 0.7],
        ["PROSPERO:", 100, 0.7]
    ]
)

if __name__ == "__main__":
    demo.launch() 