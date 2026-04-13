"""generation.chain — ReAct agent with retrieval tool for tool_use_rag.

The agent:
  1. Receives a user question.
  2. Decides whether to call the ``retrieve_documents`` tool (and with what query).
  3. May call the tool multiple times with refined queries.
  4. Synthesises a grounded answer from the retrieved passages.
  5. Returns an Answer dict with source references parsed from the agent trace.

Answer schema:
{
  "answer":           str,
  "sources":          [{"document": str, "page_or_chunk": int, "score": float, "excerpt": str}],
  "confidence":       "high" | "medium" | "low",
  "retrieved_chunks": int,
  "agent_steps":      int
}
"""

import json
import logging
import re
import time
import traceback
from typing import TypedDict

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

from config import Config

logger = logging.getLogger(__name__)

_EXCERPT_LENGTH = 220

# ReAct prompt for the agent — instructs it to use the retrieval tool greedily.
_REACT_TEMPLATE = """You are a precise research assistant for the RoboMarket domain.
Answer questions ONLY using information retrieved from the knowledge base via the tools provided.
Do NOT fabricate information. If the knowledge base does not contain the answer, say so explicitly.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question, citing the sources used (e.g. [source: filename, chunk X])

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

_PROMPT = PromptTemplate.from_template(_REACT_TEMPLATE)


class SourceRef(TypedDict):
    document: str
    page_or_chunk: int
    score: float
    excerpt: str


class Answer(TypedDict):
    answer: str
    sources: list[SourceRef]
    confidence: str
    retrieved_chunks: int
    agent_steps: int


# ── Public API ────────────────────────────────────────────────────────────────


def run_agent(
    query: str,
    tools: list[Tool],
    model: str | None = None,
) -> Answer:
    """Run the ReAct agent and return a structured Answer.

    Args:
        query:  The user question.
        tools:  List of LangChain Tools (must include retrieve_documents).
        model:  Override the default chat model from Config.

    Returns:
        An Answer dict with the grounded answer, sources, and agent metadata.
    """
    t0 = time.monotonic()

    llm = ChatOpenAI(
        model=model or Config.OPENAI_MODEL,
        temperature=0,
        openai_api_key=Config.OPENAI_API_KEY,
    )
    agent = create_react_agent(llm=llm, tools=tools, prompt=_PROMPT)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=Config.AGENT_MAX_ITERATIONS,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        verbose=False,
    )

    try:
        result = executor.invoke({"input": query})
    except Exception as exc:
        _log_error("agent", str(exc), traceback.format_exc())
        raise

    answer_text: str = result.get("output", "")
    intermediate_steps = result.get("intermediate_steps", [])
    agent_steps = len(intermediate_steps)

    sources, total_chunks = _extract_sources(intermediate_steps)
    confidence = _confidence_from_sources(sources)

    duration_ms = int((time.monotonic() - t0) * 1000)
    logger.info(json.dumps({"status": "ok", "steps": agent_steps, "confidence": confidence, "ms": duration_ms}))

    return Answer(
        answer=answer_text.strip(),
        sources=sources,
        confidence=confidence,
        retrieved_chunks=total_chunks,
        agent_steps=agent_steps,
    )


# ── Private helpers ───────────────────────────────────────────────────────────


def _extract_sources(intermediate_steps: list) -> tuple[list[SourceRef], int]:
    """Parse source references from agent intermediate steps (tool observations)."""
    sources: list[SourceRef] = []
    total_chunks = 0

    # Pattern: "[N] filename (chunk X) relevance=0.XX\ntext..."
    _passage_re = re.compile(
        r"\[(\d+)\]\s+(\S+)\s+\((?:page|chunk)\s+(-?\d+)\)\s+relevance=([\d.]+)\n(.*?)(?=\n---|\Z)",
        re.DOTALL,
    )

    for action, observation in intermediate_steps:
        if not isinstance(observation, str):
            continue
        matches = _passage_re.findall(observation)
        total_chunks += len(matches)
        for _, doc_name, loc_str, score_str, text in matches:
            loc = int(loc_str)
            score = float(score_str)
            excerpt = text.strip()[:_EXCERPT_LENGTH]
            if len(text.strip()) > _EXCERPT_LENGTH:
                excerpt += "…"
            sources.append(
                SourceRef(
                    document=doc_name,
                    page_or_chunk=loc,
                    score=round(score, 4),
                    excerpt=excerpt,
                )
            )

    # Deduplicate by (document, page_or_chunk)
    seen: set[tuple[str, int]] = set()
    unique: list[SourceRef] = []
    for s in sources:
        key = (s["document"], s["page_or_chunk"])
        if key not in seen:
            seen.add(key)
            unique.append(s)

    return unique, total_chunks


def _confidence_from_sources(sources: list[SourceRef]) -> str:
    if not sources:
        return "low"
    top_score = max(s["score"] for s in sources)
    if top_score >= Config.CONFIDENCE_HIGH:
        return "high"
    if top_score >= Config.CONFIDENCE_MEDIUM:
        return "medium"
    return "low"


def _log_error(stage: str, message: str, tb: str) -> None:
    logger.error(json.dumps({"status": "error", "stage": stage, "message": message, "traceback": tb}))
