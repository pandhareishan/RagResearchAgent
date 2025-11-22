from fastapi import FastAPI
from pydantic import BaseModel
from ..agent.agent import ResearchAgent, AgentConfig

app = FastAPI(title="RAG Research Agent")

agent = ResearchAgent(AgentConfig())

class ChatReq(BaseModel):
    query: str

@app.post("/chat")
def chat(req: ChatReq):
    return agent.answer(req.query)

@app.get("/health")
def health():
    return {"status": "ok"}
