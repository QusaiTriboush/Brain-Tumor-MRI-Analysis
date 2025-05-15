# qwen_vl_utils.py
from PIL import Image

def process_vision_info(messages):
    image_inputs = []
    video_inputs = []

    for msg in messages:
        for content in msg["content"]:
            if content["type"] == "image":
                image = Image.open(content["image"]).convert("RGB")
                image_inputs.append(image)
            elif content["type"] == "video":
                # Placeholder: no videos used now
                video_inputs.append(None)
    
    return image_inputs, video_inputs
