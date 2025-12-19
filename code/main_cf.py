import uvicorn
from app_cf import app

if __name__ == "__main__":
    uvicorn.run("main_cf:app", host="localhost" , port=8089, reload=True)