import uvicorn
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

@app.get("/")
def home_page():
    return {"Hello": "Pupu"}


@app.post("/items")
async def read_item(id: int, image: UploadFile = File(...)):
    return {"item_id": id, "item_image_name": image.filename}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000) 