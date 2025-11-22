from typing import Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import wikipedia
import os
from uuid import uuid4

def wiki_search_tool(query: str, sentences: int = 3) -> Dict[str, Any]:
    try:
        page_title = wikipedia.search(query, results=1)
        if not page_title:
            return {"tool": "wiki_search", "result": "No results"}
        page = wikipedia.page(page_title[0], auto_suggest=False)
        summary = wikipedia.summary(page.title, sentences=sentences)
        return {"tool": "wiki_search", "title": page.title, "url": page.url, "summary": summary}
    except Exception as e:
        return {"tool": "wiki_search", "error": str(e)}

def plot_csv_tool(csv_text: str, x: str, y: str) -> Dict[str, Any]:
    # Create a plot from CSV text and save it
    df = pd.read_csv(pd.compat.StringIO(csv_text))
    plt.figure()
    df.plot(x=x, y=y, kind="line")
    outdir = "plots"
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, f"plot_{uuid4().hex[:8]}.png")
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()
    return {"tool": "plot_csv", "path": outfile, "columns": list(df.columns)}

TOOLS = {
    "wiki_search": wiki_search_tool,
    "plot_csv": plot_csv_tool,
}
