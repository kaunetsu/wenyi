"""Render the saved translator's afterword as cross-format back matter."""

from __future__ import annotations

import os
import posixpath
import zipfile
from html import escape
from urllib.parse import unquote, urlsplit

from bs4 import BeautifulSoup, Tag

from ..afterword import TranslatorAfterword
from .about import rootfile_path

AFTERWORD_FILENAME = "wenyi-translator-afterword.xhtml"


def afterword_xhtml(afterword: TranslatorAfterword, lang: str) -> bytes:
    """Return one standalone XHTML back-matter document."""
    paragraphs = "".join(f"<p>{escape(paragraph)}</p>" for paragraph in afterword.paragraphs)
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f'<html xmlns="http://www.w3.org/1999/xhtml" '
        f'xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="{escape(lang)}">'
        f"<head><title>{escape(afterword.title)}</title>"
        "<style>body{line-height:1.75;margin:5%;}h1{text-align:center;}"
        "p{text-indent:2em;margin:0.8em 0;}</style></head>"
        f'<body><section epub:type="afterword"><h1>{escape(afterword.title)}</h1>'
        f"{paragraphs}</section></body></html>"
    ).encode("utf-8")


def _unique_entry(existing_names: set[str], opf_path: str) -> tuple[str, str]:
    opf_dir = posixpath.dirname(opf_path)
    stem, ext = posixpath.splitext(AFTERWORD_FILENAME)
    suffix = 0
    while True:
        filename = AFTERWORD_FILENAME if suffix == 0 else f"{stem}-{suffix}{ext}"
        entry = posixpath.join(opf_dir, filename) if opf_dir else filename
        if entry not in existing_names:
            return entry, filename
        suffix += 1


def _unique_item_id(soup: BeautifulSoup) -> str:
    existing = {
        str(item.get("id")) for item in soup.find_all("item") if isinstance(item.get("id"), str)
    }
    item_id = "wenyi-translator-afterword"
    suffix = 1
    while item_id in existing:
        item_id = f"wenyi-translator-afterword-{suffix}"
        suffix += 1
    return item_id


def _resolved_item_path(opf_path: str, href: str) -> str:
    path = unquote(urlsplit(href).path)
    return posixpath.normpath(posixpath.join(posixpath.dirname(opf_path), path))


def _append_nav_link(data: bytes, href: str, title: str) -> bytes:
    soup = BeautifulSoup(data, "xml")
    toc_nav: Tag | None = None
    for nav in soup.find_all("nav"):
        kind = nav.get("epub:type") or nav.get("type") or nav.get("role") or ""
        if "toc" in str(kind).split() or str(kind) == "doc-toc":
            toc_nav = nav
            break
    if toc_nav is None:
        return data
    listing = toc_nav.find(["ol", "ul"])
    if not isinstance(listing, Tag):
        listing = soup.new_tag("ol")
        toc_nav.append(listing)
    item = soup.new_tag("li")
    link = soup.new_tag("a", href=href)
    link.string = title
    item.append(link)
    listing.append(item)
    return soup.encode()


def _append_ncx_link(data: bytes, href: str, title: str) -> bytes:
    soup = BeautifulSoup(data, "xml")
    nav_map = soup.find("navMap") or soup.find("navmap")
    if not isinstance(nav_map, Tag):
        return data
    play_orders: list[int] = []
    for point in soup.find_all(["navPoint", "navpoint"]):
        value = point.get("playOrder") or point.get("playorder")
        try:
            play_orders.append(int(str(value)))
        except (TypeError, ValueError):
            pass
    point = soup.new_tag("navPoint")
    point["id"] = "wenyi-translator-afterword"
    point["playOrder"] = str(max(play_orders, default=0) + 1)
    label = soup.new_tag("navLabel")
    text = soup.new_tag("text")
    text.string = title
    label.append(text)
    content = soup.new_tag("content")
    content["src"] = href
    point.extend((label, content))
    nav_map.append(point)
    return soup.encode()


def append_afterword_page(
    epub_path: str,
    afterword: TranslatorAfterword,
    lang: str,
) -> bool:
    """Atomically append an EPUB afterword and add it to package navigation when present."""
    with zipfile.ZipFile(epub_path, "r") as source:
        try:
            opf_path = rootfile_path(source.read("META-INF/container.xml"))
        except KeyError:
            return False
        if not opf_path or opf_path not in source.namelist():
            return False

        infos = source.infolist()
        entries = {info.filename: source.read(info.filename) for info in infos}
        afterword_entry, afterword_href = _unique_entry(set(entries), opf_path)
        soup = BeautifulSoup(entries[opf_path], "xml")
        manifest = soup.find("manifest")
        spine = soup.find("spine")
        if not isinstance(manifest, Tag) or not isinstance(spine, Tag):
            return False

        item_id = _unique_item_id(soup)
        item = soup.new_tag("item")
        item["id"] = item_id
        item["href"] = afterword_href
        item["media-type"] = "application/xhtml+xml"
        manifest.append(item)
        itemref = soup.new_tag("itemref")
        itemref["idref"] = item_id
        spine.append(itemref)

        for package_item in manifest.find_all("item"):
            href = package_item.get("href")
            if not isinstance(href, str) or not href:
                continue
            resource_path = _resolved_item_path(opf_path, href)
            if resource_path not in entries:
                continue
            relative_href = posixpath.relpath(
                afterword_entry,
                posixpath.dirname(resource_path) or ".",
            )
            properties = str(package_item.get("properties") or "").split()
            media_type = str(package_item.get("media-type") or "")
            if "nav" in properties:
                entries[resource_path] = _append_nav_link(
                    entries[resource_path], relative_href, afterword.title
                )
            elif media_type == "application/x-dtbncx+xml":
                entries[resource_path] = _append_ncx_link(
                    entries[resource_path], relative_href, afterword.title
                )

        entries[opf_path] = soup.encode()

    tmp_path = epub_path + ".afterword.tmp"
    try:
        with zipfile.ZipFile(tmp_path, "w") as target:
            for info in infos:
                data = entries[info.filename]
                if info.filename == "mimetype":
                    target.writestr(info, data, zipfile.ZIP_STORED)
                else:
                    target.writestr(info, data)
            target.writestr(afterword_entry, afterword_xhtml(afterword, lang))
        os.replace(tmp_path, epub_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return True


def afterword_text(afterword: TranslatorAfterword, *, markdown: bool = False) -> str:
    """Render plain-text back matter; Markdown uses one level-one heading."""
    heading = f"# {afterword.title}" if markdown else afterword.title
    return "\n\n".join((heading, *afterword.paragraphs))


def afterword_html(afterword: TranslatorAfterword) -> str:
    """Render escaped semantic HTML for HTML and non-BabelDOC PDF output."""
    paragraphs = "".join(f"<p>{escape(paragraph)}</p>" for paragraph in afterword.paragraphs)
    return (
        '<section class="wenyi-translator-afterword" role="doc-afterword">'
        f"<h1>{escape(afterword.title)}</h1>{paragraphs}</section>"
    )
