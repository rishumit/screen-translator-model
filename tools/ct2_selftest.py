"""Starts tools/ct2_server.py with a model folder and translates two sentences.

Usage: python ct2_selftest.py <tools-dir> <model-dir>
Exit code 0 when the server is ready and returns Hebrew text.
"""
import json
import pathlib
import subprocess
import sys

tools, model = pathlib.Path(sys.argv[1]), sys.argv[2]
p = subprocess.Popen([sys.executable, "-u", str(tools / "ct2_server.py"), "--model-dir", model],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8")
ready = p.stdout.readline().strip()
print(ready)
if '"ready"' not in ready:
    sys.exit(1)
p.stdin.write(json.dumps({"texts": ["Hello, how are you?", "Open the file you saved yesterday."], "requestId": "1"}) + "\n")
p.stdin.flush()
answer = p.stdout.readline().strip()
print(answer)
p.stdin.close()
p.wait(10)
translations = json.loads(answer).get("translations") or []
ok = len(translations) == 2 and all(any("\u0590" <= ch <= "\u05FF" for ch in t) for t in translations)
sys.exit(0 if ok else 1)
