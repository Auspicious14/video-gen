import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import OUTPUT_DIR, DEV_MODE
from routers.generate import router as gen_router
from routers.status import router as status_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if DEV_MODE:
        logger.warning("DEV_MODE active — models skipped.")
    else:
        logger.info("Loading AI models (this takes a minute on first run)...")
        try:
            from services.generation_engine import load_models
            load_models()
            logger.info("Models ready.")
        except Exception as e:
            logger.exception(f"Model loading failed: {e}")
            raise  # crash fast — don't start server with broken state
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="AI Video Generation API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve finished videos directly from /videos/<job_id>.mp4
app.mount("/videos", StaticFiles(directory=str(OUTPUT_DIR)), name="videos")

app.include_router(gen_router)
app.include_router(status_router)


@app.get("/health")
def health():
    return {"status": "ok", "dev_mode": DEV_MODE}


# ── Run ───────────────────────────────────────────────────────────────────────
# uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
# Workers MUST be 1 — shared GPU state is not multiprocess-safe without Redis.