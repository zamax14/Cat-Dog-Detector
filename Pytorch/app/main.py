from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .detect import router
app = FastAPI()

origins = ['*']

app.include_router(router=router)