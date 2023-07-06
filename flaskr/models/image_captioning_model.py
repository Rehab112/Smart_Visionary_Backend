import torch
from transformers import AutoProcessor, AutoModelForCausalLM

processor = AutoProcessor.from_pretrained("microsoft/git-large-coco") # base, large
image_caption_model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-coco")
GPU = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_caption_from_image(image):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(GPU)
    generated_ids = image_caption_model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption
