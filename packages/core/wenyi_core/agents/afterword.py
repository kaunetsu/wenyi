"""Generate and critically revise a grounded translator's afterword."""

from __future__ import annotations

import json
from typing import Any

from ..afterword import TranslatorAfterword
from ..glossary.store import GlossaryTerm
from ..i18n.prompts import render
from . import prompts
from .base import Agent


class AfterwordWriter(Agent):
    """Own the two-pass afterword prompts and strict response validation."""

    def generate(
        self,
        *,
        book_title: str,
        authors: list[str],
        verified_context: str,
        book_synopsis: str,
        chapter_digests: str,
        representative_passages: str,
        style_brief: str,
        glossary_terms: list[GlossaryTerm],
    ) -> TranslatorAfterword:
        evidence = self.build_evidence(
            book_title=book_title,
            authors=authors,
            verified_context=verified_context,
            book_synopsis=book_synopsis,
            chapter_digests=chapter_digests,
            representative_passages=representative_passages,
            style_brief=style_brief,
            glossary_terms=glossary_terms,
        )
        return self.revise(evidence, self.draft(evidence))

    def build_evidence(
        self,
        *,
        book_title: str,
        authors: list[str],
        verified_context: str,
        book_synopsis: str,
        chapter_digests: str,
        representative_passages: str,
        style_brief: str,
        glossary_terms: list[GlossaryTerm],
    ) -> dict[str, str]:
        """Build the exact evidence supplied to both model calls."""
        evidence = {
            "book_title": book_title.strip() or "(unknown)",
            "authors": ", ".join(
                author.strip() for author in authors if isinstance(author, str) and author.strip()
            )
            or "(none)",
            "verified_context": verified_context.strip() or "(none)",
            "book_synopsis": book_synopsis.strip() or "(none)",
            "chapter_digests": chapter_digests.strip() or "(none)",
            "representative_passages": representative_passages.strip() or "(none)",
            "style_brief": style_brief.strip() or "(none)",
            "glossary": prompts.render_glossary(glossary_terms),
        }
        return evidence

    def draft(self, evidence: dict[str, str]) -> TranslatorAfterword:
        """Produce the first draft from the fixed evidence snapshot."""
        return self._request("afterword_system", "afterword_user", evidence, "afterword.draft")

    def revise(self, evidence: dict[str, str], draft: TranslatorAfterword) -> TranslatorAfterword:
        """Fact-check and critically revise an existing draft."""
        return self._request(
            "afterword_reviser_system",
            "afterword_reviser_user",
            {**evidence, "draft": json.dumps(draft.to_dict(), ensure_ascii=False, indent=2)},
            "afterword.revise",
        )

    def _request(
        self,
        system_template: str,
        user_template: str,
        values: dict[str, str],
        operation: str,
    ) -> TranslatorAfterword:
        system = render(system_template, src=self.src, tgt=self.tgt)
        user = render(user_template, src=self.src, tgt=self.tgt, **values)
        data = self._ask_json(system, user, operation=operation)
        return self._validate(data)

    @staticmethod
    def _validate(data: Any) -> TranslatorAfterword:
        if not isinstance(data, dict):
            raise ValueError("Afterword response must be a JSON object")
        if set(data) != {"title", "paragraphs"}:
            raise ValueError("Afterword response must contain exactly title and paragraphs")

        title = data.get("title")
        raw_paragraphs = data.get("paragraphs")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("Afterword response has no valid title")
        title = title.strip()
        if len(title) > 200:
            raise ValueError("Afterword title exceeds 200 characters")
        if not isinstance(raw_paragraphs, list):
            raise ValueError("Afterword paragraphs must be an array")
        if any(not isinstance(paragraph, str) for paragraph in raw_paragraphs):
            raise ValueError("Afterword paragraphs must contain only strings")
        paragraphs = tuple(paragraph.strip() for paragraph in raw_paragraphs if paragraph.strip())
        if len(paragraphs) < 3 or len(paragraphs) > 12:
            raise ValueError("Afterword response must contain 3 to 12 nonempty paragraphs")
        if sum(len(paragraph) for paragraph in paragraphs) > 20000:
            raise ValueError("Afterword body exceeds 20000 characters")
        return TranslatorAfterword(title=title, paragraphs=paragraphs)
