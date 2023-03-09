import uvicorn
from fastapi import FastAPI, Form
from PIL import Image
import io
from inference import generate_captions

app = FastAPI()

@app.get("/")
def home_page():
    return "Welcome to Image Captioning. Please go to /predict to test your images."

@app.post("/predict")
async def read_image(url: str = Form(...)):

    caption = generate_captions(url)
    print(f"Done {'*'*50}")
    return {"caption": caption}

if __name__ == '__main__':
    uvicorn.run("app:app", host='127.0.0.1', port=8000, reload=True) 