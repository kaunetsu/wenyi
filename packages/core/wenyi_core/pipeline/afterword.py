"""Generate, fingerprint and persist the optional translator's afterword."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from ..afterword import (
    AFTERWORD_ARTIFACT,
    AFTERWORD_DRAFT_ARTIFACT,
    TranslatorAfterword,
    afterword_content_digest,
    validate_afterword_content,
)
from ..i18n.resources import prompt_fingerprint
from ..ingest.models import KIND_TEXT
from ..llm.routing import inference_snapshot
from ..storage.protocol import STATUS_DONE, Storage

if TYPE_CHECKING:
    from .runtime import PipelineRuntime

_MAX_DIGESTS = 12
_MAX_DIGEST_CHARS = 12000
_MAX_GLOSSARY_TERMS = 80
_MAX_PASSAGES = 12
_MAX_PASSAGE_CHARS = 400


def _json_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class AfterwordService:
    """Create one reusable afterword from final formal translation state."""

    def __init__(self, runtime: PipelineRuntime):
        self._runtime = runtime

    def generate(
        self,
        store: Storage,
        *,
        progress=None,
    ) -> TranslatorAfterword:
        manifest = store.load_manifest()
        pending = [
            entry.get("index")
            for entry in manifest.get("chapters", [])
            if isinstance(entry, dict) and entry.get("status") != STATUS_DONE
        ]
        if pending:
            joined = ", ".join(str(index) for index in pending[:10])
            suffix = "…" if len(pending) > 10 else ""
            raise ValueError(
                "Translator afterword requires every chapter to be translated; "
                f"pending chapters: {joined}{suffix}"
            )

        chapters = [store.load_chapter(entry["index"]) for entry in manifest["chapters"]]
        analysis = store.load_analysis() or {}
        terms = store.all_terms()
        writer = self._runtime.afterword_writer
        evidence = writer.build_evidence(
            book_title=str(manifest.get("title") or ""),
            authors=self._authors(manifest),
            verified_context=self._runtime.config.pipeline.translator_afterword_context,
            book_synopsis=str(analysis.get("book_synopsis") or ""),
            chapter_digests=self._representative_digests(chapters),
            representative_passages=self._representative_passages(chapters),
            style_brief=self._runtime.analyzer.style_brief(analysis),
            glossary_terms=self._selected_terms(terms),
        )
        fingerprint = self._fingerprint(chapters, evidence)
        existing = store.read_artifact(AFTERWORD_ARTIFACT)
        if (
            isinstance(existing, dict)
            and existing.get("version") == 1
            and existing.get("fingerprint") == fingerprint
        ):
            try:
                saved = validate_afterword_content(existing.get("content"))
            except ValueError:
                pass
            else:
                store.log_event("translator_afterword_skipped", reason="already_completed")
                return saved

        if progress:
            progress(0, 0, "Generating translator's afterword…")
        store.log_event(
            "translator_afterword_started",
            content_digest=fingerprint["content_digest"],
        )
        draft_artifact = store.read_artifact(AFTERWORD_DRAFT_ARTIFACT)
        draft: TranslatorAfterword | None = None
        if (
            isinstance(draft_artifact, dict)
            and draft_artifact.get("version") == 1
            and draft_artifact.get("fingerprint") == fingerprint
        ):
            try:
                draft = validate_afterword_content(draft_artifact.get("content"))
            except ValueError:
                pass
        if draft is None:
            draft = writer.draft(evidence)
            store.write_artifact(
                AFTERWORD_DRAFT_ARTIFACT,
                {"version": 1, "fingerprint": fingerprint, "content": draft.to_dict()},
            )
            store.log_event("translator_afterword_draft_saved")
        else:
            store.log_event("translator_afterword_draft_reused")
        afterword = writer.revise(evidence, draft)
        store.write_artifact(
            AFTERWORD_ARTIFACT,
            {
                "version": 1,
                "fingerprint": fingerprint,
                "content": afterword.to_dict(),
            },
        )
        store.log_event(
            "translator_afterword_finished",
            content_digest=fingerprint["content_digest"],
            paragraph_count=len(afterword.paragraphs),
        )
        if progress:
            progress(1, 1, "Translator's afterword generated")
        return afterword

    def _fingerprint(self, chapters, evidence: dict[str, str]) -> dict[str, Any]:
        writer = self._runtime.afterword_writer
        return {
            "version": 2,
            "content_digest": afterword_content_digest(chapters),
            "prompt_fingerprint": prompt_fingerprint(),
            "inference": inference_snapshot(
                self._runtime.config.llm,
                ("afterword.draft", "afterword.revise"),
            ),
            "context_digest": _json_digest(
                {"source_lang": writer.src, "target_lang": writer.tgt, "evidence": evidence}
            ),
        }

    @staticmethod
    def _authors(manifest: dict[str, Any]) -> list[str]:
        raw_meta = manifest.get("meta")
        meta = raw_meta if isinstance(raw_meta, dict) else {}
        raw_authors = meta.get("authors")
        if not isinstance(raw_authors, list):
            return []
        return [
            author.strip() for author in raw_authors if isinstance(author, str) and author.strip()
        ]

    @staticmethod
    def _representative_digests(chapters) -> str:
        rows = [
            (chapter.index, chapter.title, str(chapter.meta.get("source_digest") or "").strip())
            for chapter in chapters
        ]
        rows = [row for row in rows if row[2]]
        if not rows:
            return ""
        if len(rows) > _MAX_DIGESTS:
            positions = {
                round(offset * (len(rows) - 1) / (_MAX_DIGESTS - 1))
                for offset in range(_MAX_DIGESTS)
            }
            rows = [row for position, row in enumerate(rows) if position in positions]

        rendered: list[str] = []
        size = 0
        for index, title, digest in rows:
            line = f"[Chapter {index}: {title or '(untitled)'}] {digest}"
            remaining = _MAX_DIGEST_CHARS - size
            if remaining <= 0:
                break
            line = line[:remaining]
            rendered.append(line)
            size += len(line) + 1
        return "\n".join(rendered)

    @staticmethod
    def _selected_terms(terms):
        ordered = sorted(
            terms,
            key=lambda term: (
                term.first_chapter is None,
                term.first_chapter if term.first_chapter is not None else 10**9,
                term.type != "person",
                term.source,
            ),
        )
        return ordered[:_MAX_GLOSSARY_TERMS]

    @staticmethod
    def _representative_passages(chapters) -> str:
        """Sample bounded source/translation pairs across the completed book."""

        def eligible():
            for chapter in chapters:
                for segment in chapter.segments:
                    if (
                        segment.kind == KIND_TEXT
                        and segment.source.strip()
                        and (segment.target or "").strip()
                    ):
                        yield chapter, segment

        total = sum(1 for _ in eligible())
        if not total:
            return ""
        count = min(total, _MAX_PASSAGES)
        positions = (
            {round(offset * (total - 1) / (count - 1)) for offset in range(count)}
            if count > 1
            else {0}
        )
        rows = []
        for position, (chapter, segment) in enumerate(eligible()):
            if position in positions:
                rows.append(
                    f"[Chapter {chapter.index}: {chapter.title or '(untitled)'}; "
                    f"segment {segment.index}]\n"
                    f"Source: {segment.source.strip()[:_MAX_PASSAGE_CHARS]}\n"
                    f"Translation: {(segment.target or '').strip()[:_MAX_PASSAGE_CHARS]}"
                )
        return "\n\n".join(rows)
