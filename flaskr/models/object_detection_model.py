import numpy as np
import torch
import cv2
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from transformers import DPTImageProcessor, DPTForDepthEstimation
from PIL import Image




processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

processor2 = DPTImageProcessor.from_pretrained("Intel/dpt-large")
model2 = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

def detect_objects(obj, image):
    image = image.resize((256, 256))
    image = image.convert('RGB')
    inputs = processor(text=obj, images=[image] * len(obj), padding="max_length", return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    preds = outputs.logits.unsqueeze(1)
    inputs = processor2(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs2 = model2(**inputs)
        predicted_depth = outputs2.predicted_depth
    # interpolate to original size
    prediction2 = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    output2 = prediction2.squeeze().cpu().numpy()
    formatted = (output2 * 255 / np.max(output2)).astype("uint8")
    depth = Image.fromarray(formatted)
    output = ''
    for i, object in enumerate(obj):
        # Get the heat map for the prompt
        heat_map = torch.sigmoid(preds[i][0]).detach().cpu().numpy()
        # Resize the heat map to match the original image size
        heat_map_resized = cv2.resize(heat_map, (image.size[0], image.size[1]))
        # Apply a threshold to the heat map to get the mask
        mask = (heat_map_resized > 0.5).astype(np.uint8)
        # Apply the mask to the original image
        masked_depth = cv2.bitwise_and(np.array(depth), np.array(depth), mask=mask)
        # masked_depth = mask * np.array(depth)
        # Calculate the average depth value for the object in the mask
        object_depth = np.mean(masked_depth[masked_depth != 0])
        print(masked_depth[masked_depth != 0])
        print(object_depth)
        # Determine whether the object is close or far based on its depth value
        if np.count_nonzero(masked_depth) == 0:
            output += f"{object} is not found\n"
        elif object_depth > 60:
            output += f"{object} is close\n"
        else:
            output += f"{object} is far\n"
    return output





