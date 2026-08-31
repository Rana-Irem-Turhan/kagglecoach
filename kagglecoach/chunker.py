"""
Section-based chunker.

Splits markdown at H2/H3 headers. Oversized sections are sub-split at
paragraph boundaries; undersized trailing sections merge into the
previous chunk; an undersized leading title merges into the next chunk
when the result fits within `max_chars`.
"""

from __future__ import annotations

import re
from typing import List

_HEADER_RE = re.compile(r"^(#{2,3})\s+.+$", re.MULTILINE)


def _split_oversized(section: str, max_chars: int) -> List[str]:
    """Break a too-long section at paragraph boundaries; hard-slice as last resort."""
    paragraphs = [p.strip() for p in section.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    out: List[str] = []
    buffer: List[str] = []
    buffer_len = 0

    for para in paragraphs:
        if len(para) > max_chars:
            if buffer:
                out.append("\n\n".join(buffer))
                buffer, buffer_len = [], 0
            for i in range(0, len(para), max_chars):
                out.append(para[i : i + max_chars])
            continue

        joiner_len = 2 if buffer else 0
        if buffer_len + joiner_len + len(para) > max_chars:
            out.append("\n\n".join(buffer))
            buffer = [para]
            buffer_len = len(para)
        else:
            buffer.append(para)
            buffer_len += joiner_len + len(para)

    if buffer:
        out.append("\n\n".join(buffer))

    return out


def chunk_by_sections(text: str, min_chars: int, max_chars: int) -> List[str]:
    """Split markdown into section-bounded chunks."""
    if min_chars < 0 or max_chars <= 0 or min_chars >= max_chars:
        raise ValueError(
            f"Invalid chunk bounds: min={min_chars}, max={max_chars}. "
            "Need 0 <= min_chars < max_chars."
        )

    text = text.strip()
    if not text:
        return []

    headers = list(_HEADER_RE.finditer(text))

    if not headers:
        return _split_oversized(text, max_chars)

    sections: List[str] = []
    if headers[0].start() > 0:
        preamble = text[: headers[0].start()].strip()
        if preamble:
            sections.append(preamble)

    for i, match in enumerate(headers):
        start = match.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        section = text[start:end].strip()
        if section:
            sections.append(section)

    chunks: List[str] = []
    for section in sections:
        if len(section) > max_chars:
            chunks.extend(_split_oversized(section, max_chars))
        else:
            chunks.append(section)

    # Merge undersized trailing sections into the previous chunk, but only
    # if the merged result stays within max_chars.
    merged: List[str] = []
    for chunk in chunks:
        if (
            merged
            and len(chunk) < min_chars
            and len(merged[-1]) + 2 + len(chunk) <= max_chars
        ):
            merged[-1] = merged[-1] + "\n\n" + chunk
        else:
            merged.append(chunk)

    # An undersized leading chunk (usually a bare H1 title) prepends into
    # the next chunk if the result stays within max_chars.
    if (
        len(merged) >= 2
        and len(merged[0]) < min_chars
        and len(merged[0]) + 2 + len(merged[1]) <= max_chars
    ):
        merged[1] = merged[0] + "\n\n" + merged[1]
        merged.pop(0)

    return merged
