"""The shape every topic returns: one full report plus the same report broken
into named sections.

The two are never written twice. A pipeline either

  * builds its sections and lets `ReportResult.from_sections` join them into the
    full report (activities, tasks), or
  * hands over a report the model wrote whole and names the parts to carve out
    of it with `ReportResult.from_markdown` (notes),

so the full text and the sections cannot drift apart between runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .logging_setup import get_log
from .markdown import find_heading_body, normalise_title

log = get_log("report")


@dataclass
class Section:
    """One addressable part of a report.

    `key` is the stable machine name a caller asks for; `title` is what the
    reader sees. `include_heading=False` renders the body alone, which is how a
    single-section report stays byte-identical to the old un-sectioned output.
    """

    key: str
    title: str
    body: str
    level: int = 1
    bold: bool = False
    include_heading: bool = True

    @property
    def heading(self) -> str:
        text = f"**{self.title}**" if self.bold else self.title
        return f"{'#' * self.level} {text}"

    @property
    def markdown(self) -> str:
        body = (self.body or "").strip()
        if not self.include_heading:
            return body
        return f"{self.heading}\n\n{body}".strip()

    def to_dict(self) -> dict:
        return {"key": self.key, "title": self.title, "markdown": self.markdown,
                "body": (self.body or "").strip()}


@dataclass
class ReportResult:
    """What every pipeline returns. `report` is the whole thing; `sections` is
    the same content addressable by name."""

    topic: str
    report: str
    sections: list[Section] = field(default_factory=list)
    status: str = "full"
    missing_sections: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    ungrounded_figures: list[str] = field(default_factory=list)
    analytics_errors: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    extras: dict = field(default_factory=dict)

    # --- construction ------------------------------------------------------

    @classmethod
    def from_sections(cls, topic: str, sections: Iterable[Section], **kwargs) -> "ReportResult":
        """Full report = the sections joined. One source of truth."""
        sections = [s for s in sections if (s.body or "").strip()]
        report = "\n\n".join(s.markdown for s in sections).strip()
        return cls(topic=topic, report=report, sections=sections, **kwargs)

    @classmethod
    def from_markdown(cls, topic: str, report: str, wanted: Iterable[tuple[str, str]],
                      **kwargs) -> "ReportResult":
        """Full report as written, plus the named headings carved out of it.

        `wanted` is [(key, heading title)]. A heading the model omitted is
        recorded in `missing_sections` rather than faked.
        """
        sections, missing = [], []
        for key, title in wanted:
            body = find_heading_body(report, title)
            if body:
                sections.append(Section(key=key, title=title, body=body))
            else:
                missing.append(key)
                log.warning("  section %r (%r) not found in the %s report", key, title, topic)
        return cls(topic=topic, report=report.strip(), sections=sections,
                   missing_sections=missing, **kwargs)

    # --- access ------------------------------------------------------------

    def section(self, key_or_title: str) -> Section | None:
        target = normalise_title(key_or_title)
        for sec in self.sections:
            if sec.key == key_or_title or normalise_title(sec.title) == target:
                return sec
        return None

    def section_markdown(self, key_or_title: str) -> str:
        sec = self.section(key_or_title)
        return sec.markdown if sec else ""

    @property
    def section_keys(self) -> list[str]:
        return [s.key for s in self.sections]

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "status": self.status,
            "report": self.report,
            "sections": [s.to_dict() for s in self.sections],
            "missing_sections": self.missing_sections,
            "ungrounded_figures": self.ungrounded_figures,
            "analytics_errors": self.analytics_errors,
            "usage": self.usage,
            "seconds": self.seconds,
        }

    # --- output ------------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        """Write report.md, sections/<key>.md and report.json. Never raises: an
        unwritable directory costs the files, not the report."""
        out = Path(directory)
        try:
            (out / "sections").mkdir(parents=True, exist_ok=True)
            (out / "report.md").write_text(self.report, encoding="utf-8")
            for sec in self.sections:
                (out / "sections" / f"{sec.key}.md").write_text(sec.markdown, encoding="utf-8")
            (out / "report.json").write_text(
                json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            log.warning("could not write report files to %s: %s", out, exc)
        return out
