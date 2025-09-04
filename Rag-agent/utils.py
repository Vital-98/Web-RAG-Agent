import os, re, json

def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

def clean_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def file_exists(path: str) -> bool:
    return os.path.exists(path) and os.path.isfile(path)