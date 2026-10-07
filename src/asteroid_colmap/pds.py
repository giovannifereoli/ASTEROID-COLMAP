"""Minimal PDS3 label parser and Apache directory-listing reader.

Only what the Dawn FC labels need: ``KEY = VALUE`` statements, multi-line values
(parenthesised vectors, quoted strings, values starting on the next line), units in angle
brackets, ``"N/A"`` and nested OBJECT/GROUP blocks (flattened to ``OBJECT.KEY``). Parsing
stops at the first ``END`` line, so the HISTORY object appended after it is ignored.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_UNIT = re.compile(r"\s*<[^<>]*>\s*$")
_ITEM = re.compile(r'"[^"]*"|[^,]+')
_HREF = re.compile(r'href="([^"?/#]+)"', re.IGNORECASE)


def _scalar(token: str) -> Any:
    """Convert one PDS3 value token to ``int``, ``float``, ``str`` or ``None``.

    A trailing ``<unit>`` is dropped. Double-quoted strings have their whitespace
    collapsed. ``"N/A"``, ``N/A``, ``'N/A'``, ``NULL`` and ``UNK`` become ``None``. Other
    tokens are tried as ``int``, then ``float``, and otherwise returned as text.
    """
    tok = _UNIT.sub("", token.strip())
    if len(tok) >= 2 and tok[0] == tok[-1] == '"':
        text = " ".join(tok[1:-1].split())
        return None if text == "N/A" else text
    if tok in ("N/A", "'N/A'", "NULL", "UNK"):
        return None
    for cast in (int, float):
        try:
            return cast(tok)
        except ValueError:
            pass
    return tok


def _value(raw: str) -> Any:
    """Convert a raw PDS3 value; a parenthesised list becomes a tuple of scalars."""
    raw = raw.strip()
    if raw.startswith("(") and raw.endswith(")"):
        return tuple(_scalar(x) for x in _ITEM.findall(raw[1:-1]) if x.strip())
    return _scalar(raw)


def _incomplete(value: str) -> bool:
    """True if a value continues on the next line (empty, or unclosed ``(`` or quote)."""
    return value == "" or value.count("(") > value.count(")") or value.count('"') % 2 == 1


def parse_label(text: str) -> dict[str, Any]:
    """Parse PDS3 label text into a flat ``{key: value}`` dict.

    Keys inside OBJECT/GROUP blocks are prefixed with the block names, joined by dots
    (e.g. ``IMAGE.LINES``). Parenthesised values become tuples, units are dropped and
    N/A-type values become ``None``. Comments are skipped and parsing stops at the first
    ``END`` line.

    Parameters
    ----------
    text : str
        Label text.

    Returns
    -------
    dict
        Keywords in file order.
    """
    out: dict[str, Any] = {}
    stack: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        i += 1
        if stripped == "END":
            break
        if not stripped or stripped.startswith("/*") or "=" not in stripped:
            continue
        key, value = (s.strip() for s in stripped.split("=", 1))
        while _incomplete(value) and i < len(lines):
            value = f"{value} {lines[i].strip()}".strip()
            i += 1
        if key in ("OBJECT", "GROUP"):
            stack.append(value.strip('"'))
        elif key in ("END_OBJECT", "END_GROUP"):
            if stack:
                stack.pop()
        else:
            out[".".join([*stack, key])] = _value(value)
    return out


def read_label(path: str | Path) -> dict[str, Any]:
    """Read and parse a PDS3 label file (latin-1 encoded).

    Parameters
    ----------
    path : str or pathlib.Path
        Detached ``.LBL`` file.

    Returns
    -------
    dict
        See :func:`parse_label`.
    """
    return parse_label(Path(path).read_text(encoding="latin-1"))


def parse_listing(html: str) -> list[str]:
    """File names linked from an Apache ``Index of`` page.

    Parameters
    ----------
    html : str
        Page source.

    Returns
    -------
    list of str
        Sorted unique ``href`` targets, without sort links and sub-directories.
    """
    return sorted({name for name in _HREF.findall(html) if not name.startswith("?")})
