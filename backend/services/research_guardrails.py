"""Guardrails AI validation for source citations and forward-looking rate claims."""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable

from guardrails import Guard
from guardrails.validator_base import FailResult, PassResult, ValidationResult, Validator


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[`*_>#\"']", "", text)).strip().casefold()


class MacroMemorandumValidator(Validator):
    """Reject unsupported citation labels, statutory references, or rate projections."""

    _citation_patterns = (
        re.compile(r"(?i)\bCited:\s*(.+?)\)\*(?=\s|$)"),
        re.compile(r"(?m)^\s*\d+\.\s+\*\*(.+?)\*\*"),
    )
    _statute_pattern = re.compile(
        r"(?:\b(?:Act|Code|Regulation|Section|Sec\.?|Article|Art\.?)\s+|\u00a7\s+)"
        r"(?:No\.?\s*)?[A-Z]?[\w.-]*(?:\([\w.-]+\))*",
        re.IGNORECASE,
    )
    _rate_pattern = re.compile(
        r"\b(?:policy|repo|interest|federal funds|benchmark|mortgage)\s+rates?\b|\brates?\b",
        re.IGNORECASE,
    )
    _projection_pattern = re.compile(
        r"\b(?:will|would|is|are|was|were)\s+(?:likely\s+to\s+|expected\s+to\s+|projected\s+to\s+|forecast\s+to\s+)?"
        r"(?:be\s+)?(?:cut|hike|raise|lower|rise|fall|increase|decrease|decline|remain|hold|keep|kept|held|maintain|move|reach|stay|reduce|ease|tighten)\b"
        r"|\b(?:expected|projected|forecast|anticipate[sd]?)\s+to\s+"
        r"(?:be\s+)?(?:cut|hike|raise|lower|rise|fall|increase|decrease|decline|remain|hold|keep|kept|held|maintain|move|reach|stay|reduce|ease|tighten)\b"
        r"|\b(?:markets?\s+)?expect(?:s|ed)?\s+(?:policy\s+|repo\s+|interest\s+)?rates?\s+to\s+"
        r"(?:be\s+)?(?:cut|hike|raise|lower|rise|fall|increase|decrease|decline|remain|hold|keep|kept|held|maintain|move|reach|stay|reduce|ease|tighten)\b"
        r"|\b(?:policy\s+|repo\s+|interest\s+)?rates?\s+(?:may|might|could|should|likely\s+to)\s+"
        r"(?:be\s+)?(?:cut|hike|raise|lower|rise|fall|increase|decrease|decline|remain|hold|keep|kept|held|maintain|move|reach|stay|reduce|ease|tighten)\b"
        r"|\b(?:rate\s+cuts?|rate\s+hikes?)\s+(?:are\s+)?(?:likely|expected|projected|forecast)\b",
        re.IGNORECASE,
    )

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    @staticmethod
    def _iter_citations(text: str) -> Iterable[str]:
        for pattern in MacroMemorandumValidator._citation_patterns:
            yield from (match.strip() for match in pattern.findall(text))

    def _validate(self, value: Any, metadata: Dict[str, Any]) -> ValidationResult:
        memorandum = str(value)
        sources = metadata.get("sources", [])
        allowed_citations = {
            str(source.get("citation", "")).strip().casefold()
            for source in sources if source.get("citation")
        }
        evidence = _normalise(" ".join(
            " ".join(str(source.get(field, "")) for field in ("citation", "title", "publisher", "content"))
            for source in sources
        ))
        issues = []

        if not sources:
            issues.append("No retrieved authoritative source is available for citation validation.")
        for citation in self._iter_citations(memorandum):
            if citation.casefold() not in allowed_citations:
                issues.append(f"Citation is not present in retrieved sources: {citation}")

        for match in self._statute_pattern.finditer(memorandum):
            reference = _normalise(match.group(0))
            if reference and reference not in evidence:
                issues.append(f"Statutory or section reference is not verified by retrieved sources: {match.group(0)}")

        # Conservative rate forecast detector. A forecast is cleared only when its
        # normalized sentence occurs verbatim in the retrieved source evidence.
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", memorandum):
            if self._rate_pattern.search(sentence) and self._projection_pattern.search(sentence):
                claim = _normalise(sentence)
                if not claim or claim not in evidence:
                    issues.append(f"Unverified forward-looking rate projection blocked: {sentence.strip()[:240]}")

        if issues:
            return FailResult(error_message="\n".join(dict.fromkeys(issues)))
        return PassResult()

    # Keep the custom validator usable with both Guardrails 0.6 and the current
    # 0.11 API, whose base class delegates validate() to _validate().
    def validate(self, value: Any, metadata: Dict[str, Any]) -> ValidationResult:
        return self._validate(value, metadata)


def validate_research_memorandum(memorandum: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate via Guardrails and fail closed by raising on any validation failure."""
    guard = Guard().use(MacroMemorandumValidator(), on_fail="exception")
    outcome = guard.validate(memorandum, metadata={"sources": sources})
    passed = bool(getattr(outcome, "validation_passed", True))
    if not passed:
        raise ValueError("Guardrails rejected the research memorandum.")
    return {
        "passed": True,
        "validator": "macro_citations_and_rate_projections",
        "source_count": len(sources),
        "issues": [],
    }
