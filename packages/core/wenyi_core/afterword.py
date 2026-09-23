"""Pure translator-afterword model and persisted-artifact validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .review.session import content_digest
from .storage.protocol import Storage

AFTERWORD_ARTIFACT = "translator-afterword.json"
AFTERWORD_DRAFT_ARTIFACT = "translator-afterword-draft.json"


def afterword_content_digest(chapters) -> str:
    """Identify formal chapter text and titles before export-only transformations."""
    payload = [content_digest(chapters), [(chapter.index, chapter.title) for chapter in chapters]]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TranslatorAfterword:
    """Validated model output rendered as export-only back matter."""

    title: str
    paragraphs: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "paragraphs": list(self.paragraphs)}


def validate_afterword_content(content: Any) -> TranslatorAfterword:
    """Validate the stable content portion of an afterword artifact."""
    if not isinstance(content, dict):
        raise ValueError("Saved translator afterword has no valid content")
    title = content.get("title")
    paragraphs = content.get("paragraphs")
    if not isinstance(title, str) or not title.strip() or len(title.strip()) > 200:
        raise ValueError("Saved translator afterword has no valid title")
    if (
        not isinstance(paragraphs, list)
        or not paragraphs
        or any(not isinstance(paragraph, str) or not paragraph.strip() for paragraph in paragraphs)
    ):
        raise ValueError("Saved translator afterword has invalid paragraphs")
    if len(paragraphs) > 12 or sum(len(paragraph) for paragraph in paragraphs) > 20000:
        raise ValueError("Saved translator afterword exceeds its supported size")
    return TranslatorAfterword(
        title=title.strip(),
        paragraphs=tuple(paragraph.strip() for paragraph in paragraphs),
    )


def load_translator_afterword(store: Storage) -> TranslatorAfterword | None:
    """Load saved back matter only when it matches the formal translation snapshot."""
    artifact = store.read_artifact(AFTERWORD_ARTIFACT)
    if artifact is None:
        return None
    if not isinstance(artifact, dict) or artifact.get("version") != 1:
        raise ValueError("Saved translator afterword has an unsupported format")
    manifest = store.load_manifest()
    chapters = [store.load_chapter(entry["index"]) for entry in manifest["chapters"]]
    fingerprint = artifact.get("fingerprint")
    if not isinstance(fingerprint, dict) or fingerprint.get(
        "content_digest"
    ) != afterword_content_digest(chapters):
        raise ValueError(
            "Saved translator afterword is out of date with the translation; "
            "regenerate it or export with the afterword disabled"
        )
    return validate_afterword_content(artifact.get("content"))
