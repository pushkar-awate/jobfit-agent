"""Quick Groq connectivity + model check. Run: python diagnose.py"""
import os, json, urllib.request, urllib.error

key = os.environ.get("GROQ_API_KEY", "")
print("GROQ_API_KEY present:", bool(key), "| starts with:", (key[:7] + "...") if key else "(none)")


def call(url, data=None):
    req = urllib.request.Request(
        url,
        data=(json.dumps(data).encode() if data else None),
        method=("POST" if data else "GET"),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json", "User-Agent": "jobfit-agent/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:
        return None, "%s: %s" % (type(e).__name__, e)

print("\n--- 1) chat test with llama-3.3-70b-versatile ---")
code, body = call("https://api.groq.com/openai/v1/chat/completions",
                  {"model": "llama-3.3-70b-versatile",
                   "messages": [{"role": "user", "content": "say OK"}]})
print("status:", code)
print(body[:700])

print("\n--- 2) models your key can use ---")
code, body = call("https://api.groq.com/openai/v1/models")
print("status:", code)
try:
    ids = sorted(m["id"] for m in json.loads(body).get("data", []))
    print("\n".join(ids))
except Exception:
    print(body[:700])
