import os
from typing import List, Dict
from .rag import RAGStore
from .tools import TOOLS, wiki_search_tool
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

class AgentConfig(BaseModel):
    index_dir: str = os.getenv("INDEX_DIR", "data/index")
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    top_k: int = int(os.getenv("TOP_K", "4"))
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")

class ResearchAgent:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.rag = RAGStore(cfg.index_dir, cfg.embed_model)

    def _llm(self, prompt: str) -> str:
        # Use OpenAI if available; otherwise a minimal offline template.
        if self.cfg.openai_api_key:
            from openai import OpenAI
            client = OpenAI(api_key=self.cfg.openai_api_key)
            resp = client.chat.completions.create(
                model=self.cfg.openai_model,
                messages=[{"role": "system", "content": "You are a careful research assistant. Cite your sources when possible."},
                          {"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        else:
            return "LLM (offline mode): " + prompt[:800]

    def plan_tools(self, query: str) -> List[Dict]:
        plans = []
        if any(k in query.lower() for k in ["who is", "what is", "wikipedia"]):
            plans.append({"name": "wiki_search", "args": {"query": query}})
        return plans

    def answer(self, query: str) -> Dict:
        # Retrieve
        ctx = self.rag.retrieve(query, k=self.cfg.top_k)
        # Simple tool planning & execution
        tool_outputs = []
        for plan in self.plan_tools(query):
            fn = TOOLS.get(plan["name"])
            if fn:
                tool_outputs.append(fn(**plan["args"]))
        # Compose prompt
        sources = "\n\n".join([f"[{i+1}] {c['text']} (source: {c['path']})" for i, c in enumerate(ctx)])
        tools_txt = "\n\n".join([str(t) for t in tool_outputs]) or "No tools used."
        prompt = f"""
Answer the user's query using the SOURCES and any TOOL_RESULTS below. If you cite, use [#] for local sources or (wiki) for Wikipedia.

Question: {query}

SOURCES:
{sources}

TOOL_RESULTS:
{tools_txt}

Write a concise, factual answer with bullet points and cite used sources.
"""
        completion = self._llm(prompt)
        return {"answer": completion, "retrieved": ctx, "tools": tool_outputs}
