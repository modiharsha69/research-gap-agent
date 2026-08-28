"""
Research Gap Identifier Agent
==============================

Ingests a small set of research paper abstracts/summaries and produces a
structured list of potential research gaps: description, supporting papers,
evidence, priority, and a suggested future research direction.

Modes:
  1. LLM mode (preferred) — uses OpenAI or Anthropic to reason over the
     supplied abstracts. Auto-selected if OPENAI_API_KEY or
     ANTHROPIC_API_KEY is set in the environment.
  2. Heuristic mode — a keyword-based fallback that works offline, useful
     for testing or when no API key is available.

Usage:
    export OPENAI_API_KEY="sk-..."      # or export ANTHROPIC_API_KEY="..."
    python research_gap_agent.py
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# 1. Input data model
# ---------------------------------------------------------------------------

@dataclass
class Paper:
    """A single research paper supplied to the agent."""
    identifier: str                 # e.g. "Paper 1" or "Smith et al."
    summary: str                    # abstract / summary text
    title: Optional[str] = None
    author: Optional[str] = None
    year: Optional[str] = None

    def label(self) -> str:
        extra = ", ".join(p for p in [self.title, self.author, self.year] if p)
        return f"{self.identifier}" + (f" ({extra})" if extra else "")


@dataclass
class AgentInput:
    papers: List[Paper]
    research_domain: Optional[str] = None
    focus_area: Optional[str] = None


# ---------------------------------------------------------------------------
# 2. Output data model
# ---------------------------------------------------------------------------

@dataclass
class ResearchGap:
    title: str
    description: str
    supporting_papers: List[str]
    evidence: str
    priority: str                          # High / Medium / Low
    suggested_direction: str
    confidence_note: Optional[str] = None  # set when evidence is weak/ambiguous


@dataclass
class GapReport:
    gaps: List[ResearchGap]
    overall_summary: str

    def to_markdown(self) -> str:
        lines = ["Potential Research Gaps:\n"]
        for i, gap in enumerate(self.gaps, 1):
            lines.append(f"{i}. {gap.title}")
            lines.append(f"   - Gap: {gap.description}")
            lines.append(f"   - Supporting Papers: {', '.join(gap.supporting_papers)}")
            lines.append(f"   - Evidence: {gap.evidence}")
            lines.append(f"   - Priority: {gap.priority}")
            lines.append(f"   - Potential Research Direction: {gap.suggested_direction}")
            if gap.confidence_note:
                lines.append(f"   - Note: {gap.confidence_note}")
            lines.append("")
        lines.append("Overall Research Gap Summary:")
        lines.append(self.overall_summary)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. LLM backend (auto-selects OpenAI or Anthropic based on env vars)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a Research Gap Identifier Agent. You analyze a small \
set of research paper abstracts/summaries and identify potential research \
gaps: limitations, unresolved problems, or areas needing further investigation.

Rules you must follow strictly:
- Base your analysis ONLY on the information in the supplied abstracts/summaries.
  Never invent findings, methods, or limitations that are not present in the text.
- Group similar limitations mentioned across multiple papers into a single gap
  where appropriate, and list all papers that support it.
- If a possible gap is only weakly supported (mentioned vaguely, or by only one
  paper, or with ambiguous wording), still include it but add a confidence_note
  explaining why the support is limited.
- Prioritize gaps as High if mentioned by multiple papers or central to the
  stated focus area/domain, Medium if mentioned clearly by one paper, Low if
  speculative or only implied.
- Output ONLY valid JSON, no prose outside the JSON.
"""

JSON_SCHEMA_INSTRUCTIONS = """Return ONLY valid JSON with this exact shape:

{
  "gaps": [
    {
      "title": "short gap title",
      "description": "1-3 sentence description of the gap",
      "supporting_papers": ["Paper 1", "Paper 3"],
      "evidence": "why this counts as a gap, referencing what those papers reported",
      "priority": "High",
      "suggested_direction": "a concrete, actionable future research direction",
      "confidence_note": null
    }
  ],
  "overall_summary": "2-4 sentence synthesis of the common themes across gaps"
}
"""


def _build_user_prompt(agent_input: AgentInput) -> str:
    parts = []
    if agent_input.research_domain:
        parts.append(f"Research Domain: {agent_input.research_domain}")
    if agent_input.focus_area:
        parts.append(f"Optional Focus Area: {agent_input.focus_area}")

    parts.append("\nSupplied Papers:")
    for p in agent_input.papers:
        parts.append(f"\n{p.label()}:\n{p.summary.strip()}")

    parts.append("\n" + JSON_SCHEMA_INSTRUCTIONS)
    return "\n".join(parts)


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """Calls whichever LLM provider has an API key set. Raises if none found."""
    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content

    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929"),
            max_tokens=2000,
            temperature=0.2,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return resp.content[0].text

    raise RuntimeError("No LLM API key found (set OPENAI_API_KEY or ANTHROPIC_API_KEY).")


def _parse_llm_json(raw: str) -> GapReport:
    # Strip markdown code fences if the model added them anyway.
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    data = json.loads(cleaned)

    gaps = [
        ResearchGap(
            title=g.get("title", "Untitled Gap"),
            description=g.get("description", ""),
            supporting_papers=g.get("supporting_papers", []),
            evidence=g.get("evidence", ""),
            priority=g.get("priority", "Medium"),
            suggested_direction=g.get("suggested_direction", ""),
            confidence_note=g.get("confidence_note"),
        )
        for g in data.get("gaps", [])
    ]
    return GapReport(gaps=gaps, overall_summary=data.get("overall_summary", ""))


# ---------------------------------------------------------------------------
# 4. Offline heuristic fallback (no API key required)
# ---------------------------------------------------------------------------

GAP_KEYWORDS = {
    "Limited Dataset Diversity": [
        "small dataset", "limited dataset", "dataset size", "dataset diversity",
        "small sample", "sample size", "single-site", "single site",
    ],
    "Limited Generalization Across Settings": [
        "generaliz", "across hospitals", "across institutions",
        "different populations", "external validation", "multiple sites",
    ],
    "Limited Real-World Validation": [
        "real-world", "real world", "clinical setting", "limited testing",
        "deployment", "prospective study", "field testing",
    ],
    "Model Interpretability": [
        "interpretab", "explainab", "black box", "black-box", "transparency",
    ],
    "Reproducibility Concerns": [
        "reproducib", "replicat", "code not available", "not publicly available",
    ],
    "Scalability / Computational Cost": [
        "scalab", "computational cost", "resource intensive", "expensive to train",
    ],
    "Bias and Fairness": [
        "bias", "fairness", "demographic", "underrepresented",
    ],
}


def _heuristic_analysis(agent_input: AgentInput) -> GapReport:
    matches: dict[str, List[str]] = {label: [] for label in GAP_KEYWORDS}

    for paper in agent_input.papers:
        text = paper.summary.lower()
        for label, keywords in GAP_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                matches[label].append(paper.label())

    gaps: List[ResearchGap] = []
    for label, papers_hit in matches.items():
        if not papers_hit:
            continue
        priority = "High" if len(papers_hit) >= 2 else "Medium"
        confidence_note = (
            None if len(papers_hit) >= 2
            else "Mentioned by only one paper in the supplied set; treat as a "
                 "preliminary/insufficiently supported gap until confirmed elsewhere."
        )
        gaps.append(
            ResearchGap(
                title=label,
                description=f"The supplied material suggests a gap related to {label.lower()}.",
                supporting_papers=papers_hit,
                evidence=f"Keyword-based match found relevant language in: {', '.join(papers_hit)}.",
                priority=priority,
                suggested_direction=f"Future work could specifically address {label.lower()} "
                                     f"through targeted studies or expanded evaluation.",
                confidence_note=confidence_note,
            )
        )

    if not gaps:
        gaps.append(
            ResearchGap(
                title="No Clear Gap Identified",
                description="No common limitation keywords were detected across the supplied summaries.",
                supporting_papers=[p.label() for p in agent_input.papers],
                evidence="Heuristic keyword scan found no overlapping limitation terms.",
                priority="Low",
                suggested_direction="Provide more detailed abstracts/summaries, or supply an LLM API key "
                                     "for deeper semantic analysis.",
                confidence_note="This is an offline heuristic result, not a full semantic analysis.",
            )
        )

    top = sorted(matches.items(), key=lambda kv: len(kv[1]), reverse=True)
    top_labels = [label for label, hits in top if hits][:2]
    summary = (
        f"The supplied papers commonly indicate gaps related to {', and '.join(top_labels)}."
        if top_labels else
        "No strongly overlapping gaps were detected in the supplied material."
    )
    return GapReport(gaps=gaps, overall_summary=summary)


# ---------------------------------------------------------------------------
# 5. The agent itself
# ---------------------------------------------------------------------------

class ResearchGapAgent:
    """Analyzes supplied paper abstracts and identifies potential research gaps."""

    def run(self, agent_input: AgentInput) -> GapReport:
        if not agent_input.papers:
            raise ValueError("At least one paper abstract/summary is required.")

        try:
            user_prompt = _build_user_prompt(agent_input)
            raw = _call_llm(SYSTEM_PROMPT, user_prompt)
            return _parse_llm_json(raw)
        except RuntimeError:
            # No API key available — fall back to offline heuristic mode.
            return _heuristic_analysis(agent_input)
        except (json.JSONDecodeError, KeyError):
            # LLM returned malformed output — fall back rather than crash.
            return _heuristic_analysis(agent_input)


# ---------------------------------------------------------------------------
# 6. Demo entry point (uses the example from the assignment)
# ---------------------------------------------------------------------------

def _example_input() -> AgentInput:
    return AgentInput(
        research_domain="Machine Learning in Healthcare",
        papers=[
            Paper(
                identifier="Paper 1",
                summary="A machine learning model for disease prediction, with "
                        "limitations related to small datasets and limited testing "
                        "across hospitals.",
            ),
            Paper(
                identifier="Paper 2",
                summary="A deep learning approach for medical image analysis, "
                        "reporting limitations related to dataset diversity and "
                        "model generalization.",
            ),
            Paper(
                identifier="Paper 3",
                summary="An AI-based diagnostic prediction system, with limitations "
                        "involving interpretability and limited real-world validation.",
            ),
        ],
    )


if __name__ == "__main__":
    agent = ResearchGapAgent()
    report = agent.run(_example_input())
    print(report.to_markdown())
