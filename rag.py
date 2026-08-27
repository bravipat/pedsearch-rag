"""
rag.py -- Retrieval-Augmented Generation over a folder of .txt/.md files.

Pipeline
--------
1. Load every .txt/.md file under a docs folder and split it into ~500
   character chunks with a bit of overlap.
2. Embed each chunk with OpenAI embeddings and build an in-memory FAISS
   index (cosine similarity via normalized inner product).
3. answer(question) embeds the question, retrieves the top-k most similar
   chunks, and asks Claude to answer using ONLY those chunks.
4. If Claude can't find the answer in the retrieved chunks, it says so and
   falls back to Anthropic's built-in web_search tool to answer from the
   internet, clearly labeling the answer as coming from the web.

Install
-------
    pip install anthropic openai faiss-cpu numpy python-dotenv

Environment
-----------
Put these in a .env file next to this script (or export them):
    ANTHROPIC_API_KEY=sk-ant-...
    OPENAI_API_KEY=sk-...

Quick use
---------
    from rag import RagIndex

    rag = RagIndex(docs_dir="./docs")
    rag.build()
    result = rag.answer("What is ...?")
    print(result["answer"])
    for src in result["sources"]:
        print(src["file"], src["chunk_id"])

Or from the command line:
    python rag.py ./docs "What is ...?"
"""

import os
import glob
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from openai import OpenAI
import anthropic

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------- config ---

CHUNK_SIZE = 500                             # target characters per chunk
CHUNK_OVERLAP = 75                           # characters of overlap between chunks
TOP_K = 4                                    # number of chunks to retrieve
EMBEDDING_MODEL = "text-embedding-3-small"   # OpenAI embedding model, 1536-dim
CLAUDE_MODEL = "claude-sonnet-5"             # answer-generation model
MAX_TOOL_TURNS = 4                           # safety cap on the tool-calling loop


@dataclass
class Chunk:
    text: str
    file: str
    chunk_id: int


class RagIndex:
    def __init__(
        self,
        docs_dir: str,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        top_k: int = TOP_K,
        embedding_model: str = EMBEDDING_MODEL,
        claude_model: str = CLAUDE_MODEL,
    ):
        self.docs_dir = docs_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.embedding_model = embedding_model
        self.claude_model = claude_model

        self.chunks: List[Chunk] = []
        self.index: Optional[faiss.Index] = None

        self._openai = OpenAI()                   # reads OPENAI_API_KEY from env
        self._anthropic = anthropic.Anthropic()    # reads ANTHROPIC_API_KEY from env

    # ---------------------------------------------------------- loading ---

    def _load_files(self) -> List[Dict[str, str]]:
        patterns = ["**/*.txt", "**/*.md"]
        paths = []
        for p in patterns:
            paths.extend(glob.glob(os.path.join(self.docs_dir, p), recursive=True))

        docs = []
        for path in sorted(set(paths)):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                if text.strip():
                    docs.append({"file": os.path.relpath(path, self.docs_dir), "text": text})
            except OSError:
                continue
        return docs

    def _chunk_text(self, text: str) -> List[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        chunks = []
        start = 0
        step = max(1, self.chunk_size - self.chunk_overlap)
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start += step
        return chunks

    # --------------------------------------------------------- embedding ---

    def _embed(self, texts: List[str]) -> np.ndarray:
        vectors = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self._openai.embeddings.create(model=self.embedding_model, input=batch)
            vectors.extend([d.embedding for d in resp.data])
        arr = np.array(vectors, dtype="float32")
        faiss.normalize_L2(arr)   # normalize so inner product == cosine similarity
        return arr

    # -------------------------------------------------------------- build ---

    def build(self) -> None:
        docs = self._load_files()
        if not docs:
            raise FileNotFoundError(f"No .txt or .md files found under {self.docs_dir!r}")

        self.chunks = []
        for doc in docs:
            for i, piece in enumerate(self._chunk_text(doc["text"])):
                self.chunks.append(Chunk(text=piece, file=doc["file"], chunk_id=i))

        vectors = self._embed([c.text for c in self.chunks])
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)

        print(f"Indexed {len(self.chunks)} chunks from {len(docs)} files.")

    # ------------------------------------------------------------ retrieve ---

    def retrieve(self, question: str) -> List[Chunk]:
        if self.index is None:
            raise RuntimeError("Call build() before retrieve()/answer().")
        q_vec = self._embed([question])
        _, idxs = self.index.search(q_vec, self.top_k)
        return [self.chunks[i] for i in idxs[0] if i != -1]

    # -------------------------------------------------------------- answer ---
    #
    # answer() runs a real Anthropic tool-calling loop instead of a hardcoded
    # "always retrieve, then check a sentinel string" pipeline. Claude is
    # given two tools -- search_pediatric_docs (a *client* tool: we execute
    # it ourselves against the FAISS index) and web_search (a *server* tool:
    # Anthropic executes it and returns the resolved result/citations in the
    # same turn) -- and decides per question which one(s) it actually needs,
    # possibly calling both. See the "Tool calling, MCP, and PedSearch" guide
    # for the full explanation of why this replaced the old NOT_FOUND_TOKEN
    # string-matching approach.

    SYSTEM_PROMPT = (
        "You are a pediatric health assistant. For any question about child health, "
        "call search_pediatric_docs first to check the indexed documents. "
        "If -- and only if -- the returned passages do not answer the question, call "
        "web_search to look it up online instead. Do NOT call web_search if the "
        "documents already answered the question well; use only one source per "
        "answer. You may call search_pediatric_docs more than once with a refined "
        "query if the first result isn't useful, but avoid unnecessary repeated calls. "
        "Answer concisely, using only information you actually retrieved, and mention "
        "which source(s) you used."
    )

    @staticmethod
    def _tools() -> List[Dict[str, Any]]:
        return [
            {
                "name": "search_pediatric_docs",
                "description": (
                    "Search the indexed pediatric health documents for passages "
                    "relevant to a question. Use this first for any question about "
                    "child health."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"}
                    },
                    "required": ["query"],
                },
            },
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
        ]

    def _run_search_tool(self, query: str, sources: List[Dict[str, Any]]) -> str:
        """Execute the client-side search_pediatric_docs tool call and record
        which chunks were used (deduped) so they can be shown as sources."""
        chunks = self.retrieve(query)
        for c in chunks:
            entry = {"file": c.file, "chunk_id": c.chunk_id, "excerpt": c.text[:200]}
            if entry not in sources:
                sources.append(entry)
        if not chunks:
            return "No relevant passages found in the indexed documents."
        return "\n\n".join(f"[{c.file} #{c.chunk_id}] {c.text}" for c in chunks)

    def answer(self, question: str) -> Dict[str, Any]:
        messages: List[Dict[str, Any]] = [{"role": "user", "content": question}]
        sources: List[Dict[str, Any]] = []
        web_citations: List[str] = []
        used_web_search = False

        for _ in range(MAX_TOOL_TURNS):
            response = self._anthropic.messages.create(
                model=self.claude_model,
                max_tokens=800,
                system=self.SYSTEM_PROMPT,
                tools=self._tools(),
                messages=messages,
            )

            # Web search is a *server* tool: Anthropic runs it and resolves it
            # within this same response, surfaced to us as citations on the
            # text block (and/or a server_tool_use block naming it).
            for block in response.content:
                if getattr(block, "type", None) == "text" and getattr(block, "citations", None):
                    used_web_search = True
                    for c in block.citations:
                        url = getattr(c, "url", None)
                        if url and url not in web_citations:
                            web_citations.append(url)
                elif getattr(block, "type", None) == "server_tool_use" and getattr(block, "name", None) == "web_search":
                    used_web_search = True

            # search_pediatric_docs is a *client* tool: any pending calls need
            # us to execute them and send a tool_result back before Claude
            # can continue.
            pending = [
                b for b in response.content
                if getattr(b, "type", None) == "tool_use" and getattr(b, "name", None) == "search_pediatric_docs"
            ]

            if not pending:
                answer_text = "".join(
                    b.text for b in response.content if getattr(b, "type", None) == "text"
                ).strip()
                return {
                    "answer": answer_text,
                    "used_web_search": used_web_search,
                    "sources": sources,
                    "web_citations": web_citations,
                }

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for call in pending:
                query = (call.input or {}).get("query") or question
                result_text = self._run_search_tool(query, sources)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": call.id, "content": result_text}
                )
            messages.append({"role": "user", "content": tool_results})

        # Safety valve: the loop above should normally end via a plain text
        # reply well before this, but don't loop forever if Claude keeps
        # calling tools.
        return {
            "answer": "I wasn't able to work out a confident answer to that -- please try rephrasing the question.",
            "used_web_search": used_web_search,
            "sources": sources,
            "web_citations": web_citations,
        }


if __name__ == "__main__":
    import sys

    docs_dir = sys.argv[1] if len(sys.argv) > 1 else "./docs"
    question = sys.argv[2] if len(sys.argv) > 2 else "What is this project about?"

    rag = RagIndex(docs_dir=docs_dir)
    rag.build()
    result = rag.answer(question)

    print("\nQ:", question)
    print("A:", result["answer"])

    if result["used_web_search"]:
        print("\n(Answered via web search -- not found in local docs)")
        for url in result["web_citations"]:
            print(" -", url)
    else:
        print("\nSources:")
        for s in result["sources"]:
            print(f" - {s['file']} (chunk {s['chunk_id']}): {s['excerpt']}...")
