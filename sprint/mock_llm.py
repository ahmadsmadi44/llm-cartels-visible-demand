"""[sprint tooling] Tiny OpenAI-compatible mock server for offline pipeline tests (no model, random prices)."""
import json, random, re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][-1]["content"]
        if "bid" in prompt.split("# Output Format")[-1][:2000] and "Your true value" in prompt or "valuation" in prompt and "buyer" in prompt[:400].lower():
            out = {"reflection": "", "plan_for_this_hour": "", "bid": round(random.uniform(88, 99), 2), "new_memory": "bid"}
        elif "coordination_score" in prompt:
            out = {"reasoning": "", "coordination_score": random.choice([0, 1, 2])}
        else:
            out = {"reflection": "", "plan_for_this_hour": "", "ask": round(random.uniform(90, 100), 2),
                   "new_memory": "ask", "plan_for_message": "", "message_to_sellers": "hold prices high", "scratch_pad_update": "x"}
        resp = {"id": "m", "object": "chat.completion", "model": body["model"], "created": 0,
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(out)}}],
                "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": 120, "total_tokens": len(prompt) // 4 + 120}}
        data = json.dumps(resp).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8765), H).serve_forever()
