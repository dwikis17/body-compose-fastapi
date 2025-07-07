from dotenv import load_dotenv
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from groq import Groq
import json
load_dotenv()

app = FastAPI()
client = Groq(
    api_key=os.environ["GROQ_API_KEY"]
)

class ImageRequest(BaseModel):
    base64_image: str

def get_groq_response(base64_image: str):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Analyze the image and extract the following fields: "
                        "weight, fat_percentage, fat_mass, muscle_mass, ffm, bone_mass, visceral_fat, tbw_percentage. "
                        "Return ONLY a valid JSON object with these fields as numbers, no type annotations, no extra text, and no explanations. "
                        "Example: {\"weight\": 70.5, \"fat_percentage\": 28.1, \"fat_mass\": 19.7, \"muscle_mass\": 39.9, "
                        "\"ffm\": 50.8, \"bone_mass\": 2.5, \"visceral_fat\": 10, \"tbw_percentage\": 55.2}"
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",  # Replace with a vision-capable model if needed
        messages=messages,
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    response_text = ""
    for chunk in completion:
        response_text += chunk.choices[0].delta.content or ""
    return response_text

@app.post("/extract")
async def extract(request: ImageRequest):
    result = get_groq_response(request.base64_image)
    try:
        # Try to parse the result as JSON
        parsed = json.loads(result)
        return JSONResponse(content=parsed)
    except Exception:
        # If parsing fails, return the raw string
        return JSONResponse(content={"result": result})
