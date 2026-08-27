"""
mcp_server.py -- Expose PedSearch's pediatric-document search over MCP.

This lets any MCP-aware application (Claude Desktop, Claude Code, or any
other MCP client) query the same document index that rag.py/app.py use,
without going through the Flask web page at all. It's an additive layer:
the actual retrieval and answer logic all still lives in rag.py -- this
file is just a second front door onto it.

Install (separate from the web app's requirements.txt -- this is a local/
dev tool, not something the Vercel deployment needs):
    pip install "mcp[cli]"

Run directly (stdio transport -- what Claude Desktop/Claude Code expect
for a locally-run server):
    python mcp_server.py

Install into Claude Desktop's config automatically:
    mcp install mcp_server.py --name "PedSearch"

Environment (same as app.py/rag.py):
    DOCS_DIR         -- folder of .txt/.md docs to index (defaults to ./docs
                        next to this file)
    OPENAI_API_KEY   -- required, for embeddings
    ANTHROPIC_API_KEY -- required, for ask_pediatric_docs' generated answers

IMPORTANT for stdio servers: stdout is the JSON-RPC protocol channel. Never
`print()` to stdout here -- it will corrupt the protocol stream. Log to
stderr instead (see _log below).
"""

import os
import sys
from typing import Optional

from mcp.server.fastmcp import FastMCP

from rag import RagIndex

_HERE = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.environ.get("DOCS_DIR") or os.path.join(_HERE, "docs")


def _log(message: str) -> None:
    print(message, file=sys.stderr)


mcp = FastMCP("PedSearch")

# Build the index once at startup, same defensive pattern as app.py: never
# let a bad/missing API key or docs folder crash the whole server -- report
# the error through the tools instead, so `mcp dev`/Claude Desktop can still
# connect and tell you what's wrong rather than failing to start at all.
_rag: Optional[RagIndex] = None
_BUILD_ERROR: Optional[str] = None
try:
    _log(f"[mcp_server] Building index from: {DOCS_DIR}")
    _rag = RagIndex(docs_dir=DOCS_DIR)
    _rag.build()
    _log(f"[mcp_server] Indexed {len(_rag.chunks)} chunks.")
except Exception as exc:  # noqa: BLE001 -- deliberately broad, see app.py's identical pattern
    _BUILD_ERROR = f"{type(exc).__name__}: {exc}"
    _log(f"[mcp_server] Startup failed: {_BUILD_ERROR}")


@mcp.tool()
def search_pediatric_docs(query: str) -> str:
    """Search the indexed pediatric health documents for passages relevant
    to a question. Returns the most relevant excerpts, each tagged with its
    source file and chunk number, or a message saying nothing relevant was
    found. This is the same retrieval step rag.py's own tool-calling loop
    uses internally -- here it's reachable directly by any MCP client.
    """
    if _BUILD_ERROR:
        return f"Index is not available: {_BUILD_ERROR}"
    chunks = _rag.retrieve(query)
    if not chunks:
        return "No relevant passages found in the indexed documents."
    return "\n\n".join(f"[{c.file} #{c.chunk_id}] {c.text}" for c in chunks)


@mcp.tool()
def ask_pediatric_docs(question: str) -> str:
    """Ask a full question and get a Claude-generated answer grounded in the
    indexed pediatric documents, with the same automatic web-search fallback
    used by the PedSearch web app -- this calls rag.py's answer() end to end,
    so it needs ANTHROPIC_API_KEY as well as OPENAI_API_KEY.
    """
    if _BUILD_ERROR:
        return f"Index is not available: {_BUILD_ERROR}"
    result = _rag.answer(question)
    lines = [result["answer"]]
    if result["sources"]:
        lines.append("\nSources:")
        for s in result["sources"]:
            lines.append(f"- {s['file']} (chunk {s['chunk_id']})")
    if result["web_citations"]:
        lines.append("\nWeb sources:")
        for url in result["web_citations"]:
            lines.append(f"- {url}")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")
