import os, subprocess, tempfile
from logger import get_logger

logger = get_logger("llm")

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL_NAME", "hf.co/unsloth/gemma-3n-E4B-it-GGUF:UD-Q4_K_XL")
MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "512"))

def generate_with_ollama(prompt: str) -> str:
    with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as tf:
        tf.write(prompt)
        path = tf.name
    cmd = ["ollama", "generate", OLLAMA_MODEL, "--prompt-file", path, "--max-tokens", str(MAX_TOKENS), "--temperature", "0.0"]
    logger.info(f"LLM → {' '.join(cmd[:3])} ...")
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    try: os.unlink(path)
    except Exception: pass
    if out.returncode != 0:
        logger.error(f"OLLAMA ERROR: {out.stderr[:400]}")
        return ""
    return out.stdout.strip()
