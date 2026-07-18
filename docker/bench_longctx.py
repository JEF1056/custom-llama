#!/usr/bin/env python3
import json, time, urllib.request

BASE = "http://localhost:8080"

def post(path, payload, timeout=1800):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

# Build a long, varied prompt (avoid degenerate repeats).
para = (
    "In the study of complex systems, researchers observe that emergent behavior "
    "arises from the interaction of many simple components, each following local "
    "rules without any central coordination. Section {n}: consider a network of "
    "agents indexed by position {n}, exchanging signals across noisy channels, "
    "adapting their internal state, and gradually converging toward a shared "
    "equilibrium that no single agent could have predicted in advance. "
)
chunks = []
i = 0
while sum(len(c) for c in chunks) < 520000:  # ~130k tokens of English
    chunks.append(para.format(n=i))
    i += 1
prompt = "".join(chunks) + "\n\nBriefly summarize the passage above in one sentence."

# Exact token count from the server tokenizer.
tok = post("/tokenize", {"content": prompt})
n_tok = len(tok["tokens"]) if isinstance(tok, dict) else len(tok)
print(f"prompt tokens = {n_tok}")

N_PREDICT = 256
t0 = time.time()
res = post("/completion", {
    "prompt": prompt,
    "n_predict": N_PREDICT,
    "temperature": 0.0,
    "cache_prompt": False,
    "stream": False,
})
wall = time.time() - t0

tm = res.get("timings", {})
print(json.dumps({
    "prompt_n":              tm.get("prompt_n"),
    "prompt_ms":             round(tm.get("prompt_ms", 0), 1),
    "prompt_per_second":     round(tm.get("prompt_per_second", 0), 2),
    "predicted_n":           tm.get("predicted_n"),
    "predicted_ms":          round(tm.get("predicted_ms", 0), 1),
    "predicted_per_second":  round(tm.get("predicted_per_second", 0), 2),
    "wall_s":                round(wall, 1),
}, indent=2))
