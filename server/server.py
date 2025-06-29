# main.py
import contextlib
from fastapi import FastAPI
from .sports_news_server import sports_news_server
import uvicorn
from fastapi.middleware.cors import CORSMiddleware


# Create a lifespan to manage session manager
@contextlib.asynccontextmanager
async def lifespan(server: FastAPI):
    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(sports_news_server.session_manager.run())
        yield


app = FastAPI(lifespan=lifespan)

# CORS (only needed if you’re calling from a browser front-end)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/", sports_news_server.streamable_http_app())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)