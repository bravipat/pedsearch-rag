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
import re
import sys
import markdown
from html import escape
from html.parser import HTMLParser
from flask import Flask, request, render_template_string

from rag import RagIndex

_LIST_ITEM_RE = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+")


def _ensure_blank_line_before_lists(text: str) -> str:
    """LLM answers often start a list right after a lead-in line with no
    blank line in between (e.g. "Key points:\\n- one\\n- two"). Standard
    markdown requires a blank line before a list to recognize it as one,
    so without this the dashes would render as literal text. Insert the
    blank line python-markdown needs, without disturbing existing lists.
    """
    lines = text.split("\n")
    out = []
    for line in lines:
        is_list_item = bool(_LIST_ITEM_RE.match(line))
        if is_list_item and out and out[-1].strip() != "" and not _LIST_ITEM_RE.match(out[-1]):
            out.append("")
        out.append(line)
    return "\n".join(out)

_MD_ALLOWED_TAGS = {
    "p", "br", "hr",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "strong", "b", "em", "i", "u", "s", "code", "pre",
    "ul", "ol", "li",
    "blockquote",
    "a",
    "table", "thead", "tbody", "tr", "th", "td",
}
_MD_ALLOWED_ATTRS = {"a": {"href", "title"}}
_UNSAFE_HREF_SCHEMES = ("javascript:", "data:", "vbscript:")


class _SafeHTMLSanitizer(HTMLParser):
    """Whitelist-based HTML sanitizer (stdlib only -- no extra dependency).

    Drops any tag not in _MD_ALLOWED_TAGS (keeping its inner text), strips
    disallowed attributes, and blocks javascript:/data: hrefs. Assumes the
    input is well-formed HTML (it comes straight out of python-markdown).
    """

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self._out = []

    def handle_starttag(self, tag, attrs):
        self._emit_start(tag, attrs, self_close=False)

    def handle_startendtag(self, tag, attrs):
        self._emit_start(tag, attrs, self_close=True)

    def _emit_start(self, tag, attrs, self_close):
        if tag not in _MD_ALLOWED_TAGS:
            return
        allowed = _MD_ALLOWED_ATTRS.get(tag, set())
        kept = []
        for name, value in attrs:
            if name not in allowed:
                continue
            if name == "href" and (value or "").strip().lower().startswith(_UNSAFE_HREF_SCHEMES):
                continue
            kept.append((name, value))
        attr_str = "".join(f' {n}="{escape(v or "", quote=True)}"' for n, v in kept)
        if tag == "a":
            attr_str += ' target="_blank" rel="noopener noreferrer"'
        self._out.append(f"<{tag}{attr_str}{' /' if self_close else ''}>")

    def handle_endtag(self, tag):
        if tag in _MD_ALLOWED_TAGS:
            self._out.append(f"</{tag}>")

    def handle_data(self, data):
        self._out.append(escape(data))

    def handle_entityref(self, name):
        self._out.append(f"&{name};")

    def handle_charref(self, name):
        self._out.append(f"&#{name};")

    def get_html(self) -> str:
        return "".join(self._out)


def _sanitize_html(html: str) -> str:
    parser = _SafeHTMLSanitizer()
    parser.feed(html)
    parser.close()
    return parser.get_html()


def render_markdown(text: str) -> str:
    """Convert an LLM-generated markdown answer into safe, styled-ready HTML."""
    if not text:
        return ""
    html = markdown.markdown(
        _ensure_blank_line_before_lists(text),
        extensions=["extra", "sane_lists", "nl2br"],
    )
    return _sanitize_html(html)

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

HERO_IMAGE_URL = (
    "https://images.unsplash.com/photo-1758691463331-2ac00e6f676f"
    "?fm=jpg&q=75&w=1600&auto=format&fit=crop"
)

PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kids Health, Answered</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --ink: #1f3347;
    --muted: #7b93a8;
    --line: #dbeefc;
    --bg: #f4f9ff;
    --card: #ffffff;
    --accent: #4fa3e3;
    --accent-2: #7ec1ee;
    --accent-ink: #ffffff;
    --gold: #ff9466;
    --local: #1f8f6e;
    --local-bg: #e4f5ee;
    --web: #b5730a;
    --web-bg: #fff1dc;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: 'Nunito', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: var(--bg);
    color: var(--ink);
    min-height: 100vh;
  }
  .wrap { max-width: 680px; margin: 0 auto; padding: 0 20px 80px; }

  .hero {
    position: relative;
    height: 260px;
    margin: 0 0 26px;
    border-radius: 0 0 32px 32px;
    overflow: hidden;
  }
  .hero-img {
    position: absolute; inset: 0;
    width: 100%; height: 100%;
    object-fit: cover;
  }
  .hero-scrim {
    position: absolute; inset: 0;
    background: linear-gradient(180deg, rgba(15,40,60,0.12) 0%, rgba(15,40,60,0.75) 100%);
  }
  .hero-inner {
    position: relative;
    height: 100%;
    display: flex; flex-direction: column; justify-content: flex-end;
    padding: 24px 24px 22px;
    color: #fff;
  }
  .hero-inner h1 {
    font-size: 27px; font-weight: 800; margin: 0 0 6px;
    text-shadow: 0 2px 10px rgba(0,0,0,0.25);
  }
  .hero-inner p {
    margin: 0; font-size: 14px; color: #eaf6ff; max-width: 440px; line-height: 1.4;
    text-shadow: 0 1px 6px rgba(0,0,0,0.25);
  }

  .features {
    display: flex; gap: 12px; margin: -46px 0 26px;
    position: relative; z-index: 2;
  }
  .feature {
    flex: 1; background: var(--card); border: 1px solid var(--line);
    border-radius: 16px; padding: 14px 10px; text-align: center;
    box-shadow: 0 8px 20px rgba(20,60,95,0.10);
  }
  .feature .ic {
    width: 34px; height: 34px; border-radius: 10px;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 8px;
  }
  .feature .ic svg { width: 17px; height: 17px; color: #fff; }
  .feature span { font-size: 11.5px; font-weight: 700; color: var(--ink); line-height: 1.3; display: block; }

  form {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 18px;
    box-shadow: 0 6px 20px rgba(20, 60, 95, 0.06);
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
    background: transparent;
  }
  textarea::placeholder { color: #a9beb8; color: #9fb3c6; }
  .form-row {
    display: flex;
    justify-content: flex-end;
    border-top: 1px solid var(--line);
    padding-top: 12px;
    margin-top: 8px;
  }
  button {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    color: var(--accent-ink);
    border: none;
    padding: 10px 22px;
    border-radius: 999px;
    font-size: 14.5px;
    font-family: inherit;
    font-weight: 700;
    cursor: pointer;
    transition: opacity 0.15s ease, transform 0.15s ease;
  }
  button:hover { opacity: 0.92; transform: translateY(-1px); }
  button:disabled { opacity: 0.55; cursor: default; transform: none; }

  .result { margin-top: 26px; }
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    font-weight: 750;
    padding: 5px 12px;
    border-radius: 999px;
    margin-bottom: 12px;
  }
  .badge.local { background: var(--local-bg); color: var(--local); }
  .badge.web { background: var(--web-bg); color: var(--web); }
  .badge svg { width: 13px; height: 13px; }

  .answer {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 22px;
    font-size: 15.5px;
    line-height: 1.6;
    box-shadow: 0 6px 20px rgba(20, 60, 95, 0.05);
    position: relative;
    overflow: hidden;
  }
  .answer .deco {
    position: absolute; right: -20px; top: -20px; opacity: 0.08;
    pointer-events: none;
  }

  .answer-md { position: relative; }
  .answer-md > *:first-child { margin-top: 0; }
  .answer-md > *:last-child { margin-bottom: 0; }
  .answer-md p { margin: 0 0 12px; }
  .answer-md h1, .answer-md h2, .answer-md h3, .answer-md h4 {
    color: var(--ink); font-weight: 800; line-height: 1.3;
    margin: 18px 0 8px;
  }
  .answer-md h1 { font-size: 19px; }
  .answer-md h2 { font-size: 17.5px; }
  .answer-md h3 { font-size: 16px; }
  .answer-md h4 { font-size: 15px; }
  .answer-md ul, .answer-md ol { margin: 0 0 12px; padding-left: 22px; }
  .answer-md li { margin-bottom: 5px; }
  .answer-md li:last-child { margin-bottom: 0; }
  .answer-md strong, .answer-md b { color: var(--ink); font-weight: 800; }
  .answer-md code {
    background: var(--bg); border: 1px solid var(--line); color: #2b5e8f;
    padding: 1px 6px; border-radius: 6px; font-size: 13.5px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  .answer-md pre {
    background: var(--bg); border: 1px solid var(--line); border-radius: 12px;
    padding: 12px 14px; overflow-x: auto; margin: 0 0 12px;
  }
  .answer-md pre code { background: none; border: none; padding: 0; }
  .answer-md blockquote {
    margin: 0 0 12px; padding: 4px 14px; border-left: 3px solid var(--accent-2);
    color: var(--muted); font-style: italic;
  }
  .answer-md a { color: var(--accent); text-decoration: none; font-weight: 600; }
  .answer-md a:hover { text-decoration: underline; }
  .answer-md hr { border: none; border-top: 1px solid var(--line); margin: 16px 0; }
  .answer-md table {
    width: 100%; border-collapse: collapse; margin: 0 0 12px; font-size: 14px;
  }
  .answer-md th, .answer-md td {
    border: 1px solid var(--line); padding: 6px 10px; text-align: left;
  }
  .answer-md th { background: var(--bg); font-weight: 800; }

  .sources {
    margin-top: 14px;
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 16px 20px;
  }
  .sources h3 {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11.5px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
    margin: 0 0 10px;
    font-weight: 800;
  }
  .sources h3 svg { width: 13px; height: 13px; }
  .sources ul { margin: 0; padding-left: 0; list-style: none; }
  .sources li {
    display: flex; gap: 8px;
    font-size: 13.5px; line-height: 1.5;
    margin-bottom: 10px; color: #52708a;
  }
  .sources li:last-child { margin-bottom: 0; }
  .sources li svg { width: 14px; height: 14px; flex: none; margin-top: 2px; color: var(--accent); }
  .sources li b { color: var(--ink); }
  .sources a { color: var(--accent); text-decoration: none; font-weight: 600; }
  .sources a:hover { text-decoration: underline; }

  .empty {
    display: flex; align-items: center; gap: 8px;
    color: var(--muted); font-size: 14px; margin-top: 14px;
  }
  .empty svg { width: 16px; height: 16px; flex: none; }

  .empty-state {
    display: flex; align-items: center; gap: 14px;
    background: var(--card); border: 1px dashed var(--line); border-radius: 18px;
    padding: 18px; margin-top: 22px; color: var(--muted); font-size: 13.5px; line-height: 1.5;
  }
  .empty-state .ic {
    width: 40px; height: 40px; border-radius: 12px; flex: none;
    background: var(--local-bg); display: flex; align-items: center; justify-content: center;
  }
  .empty-state .ic svg { width: 20px; height: 20px; color: var(--accent); }

  .setup-error {
    background: #fff2ec;
    border: 1px solid #ffcdb8;
    color: #8a3a1f;
    border-radius: 20px;
    padding: 18px 20px;
    font-size: 13.5px;
    line-height: 1.6;
    margin-bottom: 24px;
  }
  .setup-error b { display: block; margin-bottom: 4px; font-size: 14px; }
  .setup-error code {
    background: #ffe0d0; padding: 1px 6px; border-radius: 6px; font-size: 12.5px;
  }

  .disclaimer {
    display: flex; align-items: flex-start; gap: 10px;
    background: #fff9e6;
    border: 1px solid #f5dd8f;
    color: #6b5711;
    border-radius: 16px;
    padding: 13px 16px;
    font-size: 12.5px;
    line-height: 1.55;
    margin-bottom: 20px;
  }
  .disclaimer svg { width: 17px; height: 17px; flex: none; margin-top: 1px; color: #c99a1e; }
  .disclaimer b { color: #4d3f0c; }

  footer.note {
    display: flex; align-items: center; justify-content: center; gap: 6px;
    text-align: center;
    color: var(--muted);
    font-size: 12px;
    margin-top: 36px;
  }
  footer.note svg { width: 13px; height: 13px; color: var(--gold); flex: none; }
</style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <img class="hero-img" src="{{ hero_image_url }}" alt="A doctor examining a young boy while his mother looks on" loading="eager">
      <div class="hero-scrim"></div>
      <div class="hero-inner">
        <h1>Kids Health, Answered</h1>
        <p>Get clear, trustworthy answers &mdash; from your own guides, or the web when needed.</p>
      </div>
    </div>

    <div class="features">
      <div class="feature">
        <div class="ic"><svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg></div>
        <span>Grounded in your guides</span>
      </div>
      <div class="feature">
        <div class="ic"><svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg></div>
        <span>Web search backup</span>
      </div>
      <div class="feature">
        <div class="ic"><svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/></svg></div>
        <span>Fast, friendly answers</span>
      </div>
    </div>

    <div class="disclaimer">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
      <div><b>Heads up:</b> this is a personal, hobby project, not a medical product. Answers may be incomplete or wrong. Please don't use it for treatment decisions &mdash; always check with your child's doctor or another qualified professional.</div>
    </div>

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
      <textarea name="question" placeholder="Ask a question..." {{ 'disabled' if startup_error else '' }}>{{ question or '' }}</textarea>
      <div class="form-row">
        <button type="submit" id="ask-btn" {{ 'disabled' if startup_error else '' }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 2 11 13"/><path d="M22 2 15 22l-4-9-9-4 20-7z"/></svg>
          Ask
        </button>
      </div>
    </form>

    {% if error %}
      <p class="empty">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
        Something went wrong answering that: {{ error }}
      </p>
    {% endif %}

    {% if result %}
      <div class="result">
        <span class="badge {{ 'web' if result.used_web_search else 'local' }}">
          {% if result.used_web_search %}
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>
          Answered via internet search
          {% else %}
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
          Answered from your documents
          {% endif %}
        </span>
        <div class="answer">
          <svg class="deco" width="110" height="110" viewBox="0 0 24 24" fill="none" stroke="#4fa3e3" stroke-width="1.5"><path d="M12 21s-7.5-4.6-10-9.3C.5 8.4 2 4.8 5.5 4c2-.5 4 .4 5 2.2C11.5 4.4 13.5 3.5 15.5 4 19 4.8 20.5 8.4 19 11.7 16.5 16.4 12 21 12 21z"/></svg>
          <div class="answer-md">{{ result.answer_html | safe }}</div>
        </div>

        {% if result.used_web_search %}
          {% if result.web_citations %}
          <div class="sources">
            <h3>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>
              Web sources
            </h3>
            <ul>
            {% for url in result.web_citations %}
              <li>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>
                <a href="{{ url }}" target="_blank" rel="noopener">{{ url }}</a>
              </li>
            {% endfor %}
            </ul>
          </div>
          {% endif %}
        {% elif result.sources %}
          <div class="sources">
            <h3>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
              Sources
            </h3>
            <ul>
            {% for s in result.sources %}
              <li>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/></svg>
                <span><b>{{ s.file }}</b> &mdash; {{ s.excerpt }}...</span>
              </li>
            {% endfor %}
            </ul>
          </div>
        {% endif %}
      </div>
    {% elif not startup_error and not error %}
      <div class="empty-state">
        <div class="ic"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></svg></div>
        <div>Not sure what to ask? Try things like &ldquo;What are signs of an ear infection?&rdquo; or &ldquo;When should I call the doctor for a rash?&rdquo;</div>
      </div>
    {% endif %}

    <footer class="note">
      <svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2 15 9l7 1-5 5 1.5 7L12 18.5 5.5 22 7 15 2 10l7-1 3-7z"/></svg>
      Answers come from your own documents first &mdash; the web is only used as a backup.
    </footer>
  </div>

  <script>
    function onAsk(form) {
      var btn = document.getElementById('ask-btn');
      btn.disabled = true;
      btn.innerHTML = 'Thinking&hellip;';
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
                result["answer_html"] = render_markdown(result.get("answer", ""))
            except Exception as exc:  # noqa: BLE001 -- surface it in the UI instead of a 500
                error = f"{type(exc).__name__}: {exc}"
    return render_template_string(
        PAGE,
        result=result,
        error=error,
        question=question,
        docs_dir=DOCS_DIR,
        startup_error=STARTUP_ERROR,
        hero_image_url=HERO_IMAGE_URL,
    )


@app.route("/healthz")
def healthz():
    if STARTUP_ERROR:
        return {"status": "error", "detail": STARTUP_ERROR}, 500
    return {"status": "ok", "chunks": len(rag.chunks) if rag else 0}


if __name__ == "__main__":
    app.run(debug=True, port=5000)
