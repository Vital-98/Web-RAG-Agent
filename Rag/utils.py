import datetime

def log(message, level="INFO"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def format_results(results):
    formatted = ""
    for i, res in enumerate(results, 1):
        formatted += f"{i}. {res['title']}\n   URL: {res['url']}\n   Snippet: {res['snippet'][:100]}...\n\n"
    return formatted