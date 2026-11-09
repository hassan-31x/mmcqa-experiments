"""LLM translation through OpenRouter (a paid model for now; a local model replaces it later).

Every call is cached on disk by (model, direction, text), so reruns cost nothing and results are reproducible.
"""

import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from local_analysis.judge import api_key   # noqa: E402  (reuses the .env lookup; the key is never printed)

MODEL = "anthropic/claude-sonnet-5.5"
LANG = {"en": "English", "msa": "Modern Standard Arabic",
        "arz": "Egyptian Arabic (colloquial, the way people in Egypt actually write)",
        "ajp": "Levantine Arabic (colloquial, the way people in Syria, Lebanon, Jordan and Palestine actually write)"}
SRC_NAME = {"msa": "Modern Standard Arabic", "arz": "Egyptian Arabic", "ajp": "Levantine Arabic"}

TO_EN = ("Translate the following {src} question into English. Keep every named entity, place and object exactly as "
         "meant; do not answer the question, do not add or drop anything. Output only the English translation.\n\n{text}")
FROM_EN = ("Translate the following English answer into {dst}. Keep the meaning and the level of detail; write it the "
           "way a native speaker would naturally write it, in Arabic script. Do not add or remove information. "
           "Output only the translation.\n\n{text}")


class Translator:
    def __init__(self, cache_path, model=MODEL, env_file=None, workers=8):
        import requests
        self.model, self.key = model, api_key(env_file)
        self.http, self.lock, self.workers = requests.Session(), threading.Lock(), workers
        self.cache_path = Path(cache_path)
        self.cache, self.cost = {}, 0.0
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                self.cache[r["hash"]] = r

    def _hash(self, direction, track, text):
        return hashlib.sha1(json.dumps([self.model, direction, track, text], ensure_ascii=False).encode()).hexdigest()

    def _call(self, prompt):
        payload = {"model": self.model, "temperature": 0, "max_tokens": 500, "usage": {"include": True},
                   "messages": [{"role": "user", "content": prompt}]}
        last = None
        for attempt in range(5):
            try:
                r = self.http.post("https://openrouter.ai/api/v1/chat/completions", json=payload, timeout=120,
                                   headers={"Authorization": f"Bearer {self.key}"})
                if r.status_code in (401, 402, 403):
                    raise SystemExit(f"OpenRouter refused the request (HTTP {r.status_code}); check the key and credits.")
                r.raise_for_status()
                body = r.json()
                text = body["choices"][0]["message"]["content"].strip()
                if not text:
                    raise ValueError("empty translation")
                with self.lock:
                    self.cost += float(body.get("usage", {}).get("cost") or 0)
                return text
            except SystemExit:
                raise
            except Exception as exc:
                last = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"translation failed: {type(last).__name__}: {str(last)[:200]}")

    def translate_many(self, items, direction):
        """items: list of (key, track, text). direction: 'to_en' (track = source variety) or 'from_en' (track = target)."""
        out = {}

        def one(item):
            key, track, text = item
            h = self._hash(direction, track, text)
            if h in self.cache:
                return key, self.cache[h]["translation"]
            prompt = (TO_EN.format(src=SRC_NAME[track], text=text) if direction == "to_en"
                      else FROM_EN.format(dst=LANG[track], text=text))
            tr = self._call(prompt)
            rec = {"hash": h, "model": self.model, "direction": direction, "track": track, "source": text, "translation": tr}
            with self.lock:
                self.cache[h] = rec
                with self.cache_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return key, tr

        with ThreadPoolExecutor(self.workers) as pool:
            for key, tr in pool.map(one, items):
                out[key] = tr
        return out
