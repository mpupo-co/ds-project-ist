import uvicorn
from app_ta import app

if __name__ == "__main__":
    uvicorn.run("main_ta:app", host="localhost" , port=8088, reload=True)