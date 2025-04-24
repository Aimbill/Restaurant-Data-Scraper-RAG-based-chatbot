import random

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)...",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)..."
]

def get_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS)
    }
