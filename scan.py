from dotenv import load_dotenv
import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import Groq
load_dotenv()

app = FastAPI()
client = Groq(
    api_key=os.environ["GROQ_API_KEY"]
)

class ImageRequest(BaseModel):
    base64_image: str

def groq_stream(base64_image: str):
    messages = [
        {
            "role": "user",
            "content": [
                {
                     "type": "text",
                     "text": "analyze the image, extract the value and ONLY return this json format \n\n{\nweight: double,\nfat_mass: double,\nmuscle_mass:double,\nbone_mass:double,\nbmi:double,\nideal_body_weight:double\n}"
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
    for chunk in completion:
        yield chunk.choices[0].delta.content or ""

@app.post("/extract")
async def extract(request: ImageRequest):
    return StreamingResponse(groq_stream(request.base64_image), media_type="application/json")
