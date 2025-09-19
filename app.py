import os
import sys
import gradio as gr
import torch
from PIL import Image
import io
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Now import from src
from src.inference import load_model, generate_image
from src.config import DEFAULT_DEVICE

def run_demo():
    # Initialize the model
    device = torch.device(DEFAULT_DEVICE)
    print("Loading model...")
    pipe = load_model(use_img2img=True)
    pipe = pipe.to(device)
    print("Model loaded!")

    def generate(prompt, input_image, strength=0.8, guidance_scale=7.5, num_steps=30):
        try:
            # Convert Gradio Image to PIL Image
            if input_image is None:
                return None, "Please upload an input image"
                
            if isinstance(input_image, str):
                input_image = Image.open(input_image).convert("RGB")
            elif isinstance(input_image, np.ndarray):
                input_image = Image.fromarray(input_image).convert("RGB")
            
            # Ensure we have a PIL Image
            if not isinstance(input_image, Image.Image):
                if isinstance(input_image, str):
                    input_image = Image.open(input_image).convert('RGB')
                elif isinstance(input_image, np.ndarray):
                    input_image = Image.fromarray(input_image.astype('uint8'))
                else:
                    raise ValueError(f"Unsupported image type: {type(input_image)}")
            
            # Save to a temporary file to ensure proper format
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                input_image.save(tmp_file.name, format='PNG')
                tmp_path = tmp_file.name
            
            try:
                # Generate the image using the file path
                output_image = generate_image(
                    pipe=pipe,
                    prompt=prompt,
                    image_path=tmp_path,  # Pass file path
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_steps,
                    resolution=512  # Default resolution
                )
            finally:
                # Clean up the temporary file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            
            return output_image, "Generation complete!"
            
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            return None, f"Error: {str(e)}"

    # Create Gradio interface
    with gr.Blocks(title="Image Generation Demo") as demo:
        gr.Markdown("""
        # Image Generation Demo
        Upload an image and enter a prompt to modify it using our model.
        """)
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", type="pil")
                prompt = gr.Textbox(label="Prompt", placeholder="Describe how you want to modify the image...")
                
                with gr.Accordion("Advanced Settings", open=False):
                    strength = gr.Slider(minimum=0.1, maximum=1.0, value=0.8, step=0.05, label="Strength")
                    guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, value=7.5, step=0.5, label="Guidance Scale")
                    num_steps = gr.Slider(minimum=10, maximum=100, value=30, step=5, label="Number of Steps")
                
                submit_btn = gr.Button("Generate")
                
            with gr.Column():
                output_image = gr.Image(label="Generated Image")
                status = gr.Textbox(label="Status")
        
        submit_btn.click(
            fn=generate,
            inputs=[prompt, input_image, strength, guidance_scale, num_steps],
            outputs=[output_image, status]
        )
        
        gr.Examples(
            examples=[
                ["Make the image look like a watercolor painting", "examples/input1.jpg"],
                ["Convert to a night scene", "examples/input2.jpg"],
            ],
            inputs=[prompt, input_image],
            outputs=[output_image],
            fn=generate,
            cache_examples=False,
        )
    
    return demo

if __name__ == "__main__":
    # Create examples directory if it doesn't exist
    os.makedirs("examples", exist_ok=True)
    
    # Launch the demo
    demo = run_demo()
    demo.launch(share=True)  # Set share=True to get a public URL
