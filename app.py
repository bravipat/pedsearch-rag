"""
app.py -- Flask UI for rag.py

Local run:
    python app.py /path/to/your/docs
    (open http://localhost:5000)

Vercel deploy:
    Reads the DOCS_DIR environment variable, or falls back to a "docs"
    folder that sits next to this file and gets bundled with the deploy.
    See README notes in the project chat for setup steps.
"""

import os
import sys
from flask import Flask, request, render_template_string

from rag import RagIndex

_HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve_docs_dir() -> str:
    # Only trust argv when this file is actually run as a script
    # (so importing it under gunicorn/Vercel doesn't misread server argv).
    if __name__ == "__main__" and len(sys.argv) > 1:
        return sys.argv[1]
    return os.environ.get("DOCS_DIR") or os.path.join(_HERE, "docs")


DOCS_DIR = _resolve_docs_dir()

app = Flask(__name__)

# Build the index at startup, but never let a bad/missing API key or a
# missing docs folder crash the whole serverless function -- capture the
# error and show a friendly setup message on the page instead of a raw 500.
rag = None
STARTUP_ERROR = None
try:
    print(f"Building index from: {DOCS_DIR}")
    rag = RagIndex(docs_dir=DOCS_DIR)
    rag.build()
except Exception as exc:  # noqa: BLE001 -- deliberately broad: any startup failure is recoverable via the UI
    STARTUP_ERROR = f"{type(exc).__name__}: {exc}"
    print(f"Startup failed: {STARTUP_ERROR}")

PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ask your documents</title>
<style>
  :root {
    --ink: #16181d;
    --muted: #6b7280;
    --line: #e5e7eb;
    --bg: #f7f7f8;
    --card: #ffffff;
    --accent: #6d4aff;
    --accent-ink: #ffffff;
    --local: #12805c;
    --local-bg: #e6f6ef;
    --web: #9a6400;
    --web-bg: #fff3d9;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
    background: var(--bg);
    color: var(--ink);
    min-height: 100vh;
  }
  .wrap { max-width: 680px; margin: 0 auto; padding: 48px 20px 80px; }
  h1 { font-size: 22px; font-weight: 650; margin: 0 0 4px; }
  .sub { color: var(--muted); font-size: 13.5px; margin: 0 0 28px; }
  .sub code {
    background: #eee; padding: 1px 6px; border-radius: 4px; font-size: 12.5px;
  }
  form {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 16px;
    box-shadow: 0 1px 2px rgba(16,24,40,0.04);
  }
  textarea {
    width: 100%;
    min-height: 64px;
    resize: vertical;
    border: none;
    outline: none;
    font-size: 15.5px;
    font-family: inherit;
    color: var(--ink);
    padding: 4px 2px;
  }
  textarea::placeholder { color: #9aa0ab; }
  .form-row {
    display: flex;
    justify-content: flex-end;
    border-top: 1px solid var(--line);
    padding-top: 12px;
    margin-top: 8px;
  }
  button {
    background: var(--accent);
    color: var(--accent-ink);
    border: none;
    padding: 9px 20px;
    border-radius: 999px;
    font-size: 14.5px;
    font-weight: 600;
    cursor: pointer;
    transition: opacity 0.15s ease;
  }
  button:hover { opacity: 0.9; }
  button:disabled { opacity: 0.55; cursor: default; }

  .result { margin-top: 24px; }
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    font-weight: 650;
    padding: 4px 10px;
    border-radius: 999px;
    margin-bottom: 12px;
  }
  .badge.local { background: var(--local-bg); color: var(--local); }
  .badge.web { background: var(--web-bg); color: var(--web); }
  .badge .dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }

  .answer {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 20px;
    font-size: 15.5px;
    line-height: 1.55;
    white-space: pre-wrap;
  }

  .sources {
    margin-top: 14px;
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 14px 18px;
  }
  .sources h3 {
    font-size: 11.5px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
    margin: 0 0 8px;
    font-weight: 650;
  }
  .sources ul { margin: 0; padding-left: 18px; }
  .sources li { font-size: 13.5px; margin-bottom: 6px; color: #374151; }
  .sources li b { color: var(--ink); }
  .sources a { color: var(--accent); text-decoration: none; }
  .sources a:hover { text-decoration: underline; }

  .empty { color: var(--muted); font-size: 14px; margin-top: 8px; }

  .setup-error {
    background: #fff0f0;
    border: 1px solid #f3b4b4;
    color: #7a1f1f;
    border-radius: 14px;
    padding: 16px 18px;
    font-size: 13.5px;
    line-height: 1.6;
    margin-bottom: 20px;
  }
  .setup-error b { display: block; margin-bottom: 4px; font-size: 14px; }
  .setup-error code {
    background: #fbdada; padding: 1px 6px; border-radius: 4px; font-size: 12.5px;
  }
</style>
</head>
<body>
  <div class="wrap">
    <h1>Ask your documents</h1>
    <p class="sub">Indexed folder: <code>{{ docs_dir }}</code></p>

    {% if startup_error %}
      <div class="setup-error">
        <b>Setup needed</b>
        This app couldn't build its index or reach the API. This usually means
        <code>ANTHROPIC_API_KEY</code> and/or <code>OPENAI_API_KEY</code> aren't set yet.
        Add them under Project Settings &rarr; Environment Variables in Vercel, then redeploy.
        <br><br>
        Details: <code>{{ startup_error }}</code>
      </div>
    {% endif %}

    <form method="post" onsubmit="onAsk(this)">
      <textarea name="question" placeholder="Ask a question about your documents..." {{ 'disabled' if startup_error else '' }}>{{ question or '' }}</textarea>
      <div class="form-row">
        <button type="submit" id="ask-btn" {{ 'disabled' if startup_error else '' }}>Ask</button>
      </div>
    </form>

    {% if error %}
      <p class="empty">Something went wrong answering that: {{ error }}</p>
    {% endif %}

    {% if result %}
      <div class="result">
        <span class="badge {{ 'web' if result.used_web_search else 'local' }}">
          <span class="dot"></span>
          {{ 'Answered via internet search' if result.used_web_search else 'Answered from your documents' }}
        </span>
        <div class="answer">{{ result.answer }}</div>

        {% if result.sources %}
        <div class="sources">
          <h3>Sources</h3>
          <ul>
          {% for s in result.sources %}
            <li><b>{{ s.file }}</b> (chunk {{ s.chunk_id }}) &mdash; {{ s.excerpt }}...</li>
          {% endfor %}
          </ul>
        </div>
        {% endif %}

        {% if result.web_citations %}
        <div class="sources">
          <h3>Web sources</h3>
          <ul>
          {% for url in result.web_citations %}
            <li><a href="{{ url }}" target="_blank" rel="noopener">{{ url }}</a></li>
          {% endfor %}
          </ul>
        </div>
        {% endif %}
      </div>
    {% elif question %}
      <p class="empty">No question submitted.</p>
    {% endif %}
  </div>

  <script>
    function onAsk(form) {
      var btn = document.getElementById('ask-btn');
      btn.disabled = true;
      btn.textContent = 'Thinking...';
    }
  </script>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    question = None
    if request.method == "POST" and not STARTUP_ERROR:
        question = request.form.get("question", "").strip()
        if question:
            try:
                result = rag.answer(question)
            except Exception as exc:  # noqa: BLE001 -- surface it in the UI instead of a 500
                error = f"{type(exc).__name__}: {exc}"
    return render_template_string(
        PAGE,
        result=result,
        error=error,
        question=question,
        docs_dir=DOCS_DIR,
        startup_error=STARTUP_ERROR,
    )


@app.route("/healthz")
def healthz():
    if STARTUP_ERROR:
        return {"status": "error", "detail": STARTUP_ERROR}, 500
    return {"status": "ok", "chunks": len(rag.chunks) if rag else 0}


if __name__ == "__main__":
    app.run(debug=True, port=5000)
