import requests
import base64
import json
from pydantic import BaseModel, ValidationError
from typing import Literal
import os

class TumorReport(BaseModel):
    tumor_type: Literal['pituitary', 'glioma', 'meningioma', 'no tumor']
    confidence: float
    reason: str


API_KEY = ""
API_URL = "https://openrouter.ai/api/v1/chat/completions"
#  # حط التوكن تبعك هون
# API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-VL-3B-Instruct"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# الثابت prompt
PROMPT = (
    "You are a highly skilled neuroradiologist. Analyze the provided brain MRI image and classify it into ONE of the following tumor types:\n"
    "- **Pituitary**: Usually located in the sellar/suprasellar region. Look for well-defined midline mass.\n"
    "- **Glioma**: Infiltrative intra-axial mass within brain tissue, often irregular borders and edema.\n"
    "- **Meningioma**: Extra-axial, dural-based, well-circumscribed mass, may cause mass effect.\n"
    "- **No tumor**: ONLY if there are absolutely NO abnormalities.\n\n"
    "**Your response MUST be STRICTLY in JSON** with the following keys: `tumor_type`, `confidence`, and `reason`.\n\n"
    "**Rules to follow:**\n"
    "1. Choose ONLY the most likely diagnosis based on visible features.\n"
    "2. Your confidence MUST be a number between 0.0 and 1.0.\n"
    "3. In `reason`, cite the observed radiological signs that led you to the conclusion.\n"
    "4. Do NOT hallucinate. If unsure, lower the confidence score appropriately.\n"
    "5. NEVER default to 'pituitary' — only choose it if sellar/suprasellar signs are clearly seen.\n"
    "6. DO NOT add explanations outside the JSON output."
)



WORD_TO_CONF = {
    'very high': 0.95,
    'high': 0.85,
    'fairly high': 0.75,
    'moderate': 0.6,
    'medium': 0.5,
    'low': 0.3,
    'very low': 0.1
}



def analyze_image(image_path: str) -> dict:
    # Encode image to base64
    with open(image_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')

    payload = {
        'model': 'qwen/qwen-2.5-vl-7b-instruct:free',
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': PROMPT},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}}
                ]
            }
        ]
    }
    resp = requests.post(API_URL, headers=headers, json=payload)
    if resp.status_code != 200:
        raise Exception(f'API Error: {resp.status_code} {resp.text}')

    # Extract generated content
    data = resp.json()
    content = data['choices'][0]['message']['content'].strip()
    if content.startswith('```json'):
        content = content[7:-3].strip()

    # Parse raw JSON
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        raise Exception(f'Invalid JSON from model: {content}')

    # Coerce confidence field to float if needed
    conf = parsed.get('confidence')
    if isinstance(conf, str):
        try:
            parsed['confidence'] = float(conf)
        except ValueError:
            parsed['confidence'] = WORD_TO_CONF.get(conf.lower(), 0.0)

    # Validate with Pydantic
    try:
        parsed['tumor_type'] = parsed['tumor_type'].lower()
        report = TumorReport.model_validate(parsed)
    except ValidationError as ve:
        raise Exception(f'Validation error: {ve}')

    return report.model_dump()
