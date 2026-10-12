"""Laya's official laya.serve app, with the Router loading local checkpoints."""
import os

from laya.router import Router
from laya.serve import create_app


MODEL_PATH = os.getenv("LAYA_MODEL_PATH", "/models/laya")

router = Router(
    models={
        "english": MODEL_PATH,
        "multilingual": (MODEL_PATH, "multilingual"),
        "typed-decisions": (MODEL_PATH, "typed-decisions"),
    },
    device=os.getenv("LAYA_DEVICE", "cuda"),
    max_loaded=3,
)
router.preload(["english", "multilingual"])

app = create_app(router=router)
