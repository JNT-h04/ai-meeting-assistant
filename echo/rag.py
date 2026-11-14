"""Questions about one meeting, answered from the transcript with timestamps as citations.

LangChain pieces:
  - FAISS vector store over MiniLM embeddings (meaning: "budget" finds "costs")
  - BM25 retriever (exact words: names, numbers, "twelve fifty")
  - EnsembleRetriever fuses both rankings (reciprocal rank fusion)
Then the local LLM answers in JSON: {found, answer, cite}. Citations are checked:
a timestamp the retriever never returned is dropped, and "found": false gives an
honest "not discussed" instead of a made-up answer.
"""

import re
import time

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

from echo import llm, models, text

TOP_K = 4

SYSTEM = (
    "You answer questions about a meeting using only the transcript excerpts given. Each excerpt line "
    "starts with [mm:ss]. found = true only if the excerpts contain the answer. answer = one to three "
    "sentences, naming who said what. cite = the [mm:ss] times of the lines you used. If the excerpts "
    "don't contain the answer, set found = false and say it wasn't discussed in the parts found."
)
SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "answer": {"type": "string"},
        "cite": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
    },
    "required": ["found", "answer", "cite"],
}
_CLOCK = re.compile(r"\[(\d\d:\d\d)\]")


class MeetingIndex:
    def __init__(self, lines: list[str]):
        blocks = text.passages(lines, window=4, stride=2)
        docs = [Document(page_content=b, metadata={"i": i}) for i, b in enumerate(blocks)]
        dense = FAISS.from_documents(docs, models.embeddings(), distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
        sparse = BM25Retriever.from_documents(docs, k=TOP_K)
        self.retriever = EnsembleRetriever(
            retrievers=[sparse, dense.as_retriever(search_kwargs={"k": TOP_K})], weights=[0.4, 0.6])
        self.size = len(docs)

    def ask(self, question: str) -> dict:
        question = question.strip()
        t = time.perf_counter()
        found = self.retriever.invoke(question)[:TOP_K]
        found.sort(key=lambda d: d.metadata["i"])  # read excerpts in meeting order
        context = "\n...\n".join(d.page_content for d in found)
        out = llm.json_call(SYSTEM, f"Excerpts:\n{context}\n\nQuestion: {question}", SCHEMA, max_tokens=250)
        real = set(_CLOCK.findall(context))
        cites = [c.strip("[] ") for c in out.get("cite", [])]
        cites = list(dict.fromkeys(c for c in cites if c in real))
        return {
            "question": question,
            "found": bool(out.get("found")),
            "answer": out.get("answer", "").strip(),
            "cite": cites,
            "passages": [d.page_content for d in found],
            "seconds": round(time.perf_counter() - t, 1),
            "model": llm.name(),
        }
