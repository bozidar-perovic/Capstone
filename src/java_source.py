"""Mask Java comments and literal contents without changing source offsets.

This is a lexical scanner, not a Java parser or type resolver. Unicode escapes
are translated for recognizing boundaries only; executable text is returned
with its original spelling. String, character, and text-block delimiters remain
in place, while comments and literal contents become spaces. Physical CR/LF
characters are preserved. An unfinished comment or literal masks through EOF.
"""

from typing import Any


def _unicode_view(source):
    """Translate eligible Java Unicode escapes, recording original spans.

    Eligibility follows contiguous-backslash parity and the preceding Unicode
    escape rule. In particular, a Java string containing a literal backslash
    followed by 'u0022' must not acquire an artificial closing quote.
    Invalid escapes are retained; this helper does not diagnose Java syntax.
    """
    if "\\u" not in source:
        return source, None
    characters, spans = [], []
    position, slash_run, previous_escape = 0, 0, False
    while position < len(source):
        start = position
        character = source[position]
        escaped = False
        if character == "\\" and (previous_escape or slash_run % 2 == 0):
            end = position + 1
            while end < len(source) and source[end] == "u":
                end += 1
            digits = source[end:end + 4]
            if end > position + 1 and len(digits) == 4 and all(
                digit in "0123456789abcdefABCDEF" for digit in digits
            ):
                character = chr(int(digits, 16))
                position = end + 4
                escaped = True
        if not escaped:
            position += 1
        characters.append(character)
        spans.append((start, position))
        slash_run = slash_run + 1 if character == "\\" else 0
        previous_escape = escaped
    return "".join(characters), spans


def mask_non_code(source: Any) -> str:
    """Return source with comments and literal contents replaced by spaces.

    Handles line/block comments, quoted strings, character literals, Java text
    blocks, escaped delimiters, and Unicode escapes that introduce boundaries.
    Does not modify the input or join tokens separated by removed comments.
    Non-string input is treated as empty source, matching analyzer behavior.
    """
    if not isinstance(source, str):
        return ""
    text, spans = _unicode_view(source)
    result = list(source)

    def blank(start, end):
        if start == end:
            return
        raw_start = spans[start][0] if spans is not None else start
        raw_end = spans[end - 1][1] if spans is not None else end
        for index in range(raw_start, raw_end):
            if result[index] not in "\r\n":
                result[index] = " "

    position, length = 0, len(text)
    while position < length:
        if text.startswith("//", position):
            end = position + 2
            while end < length and text[end] not in "\r\n":
                end += 1
            blank(position, end)
            position = end
        elif text.startswith("/*", position):
            closing = text.find("*/", position + 2)
            end = length if closing == -1 else closing + 2
            blank(position, end)
            position = end
        elif text[position] in "\"'":
            delimiter = '"""' if text.startswith('"""', position) else (
                text[position]
            )
            content_start = position + len(delimiter)
            end = content_start
            while end < length:
                if text[end] == "\\":
                    end = min(end + 2, length)
                elif text.startswith(delimiter, end):
                    break
                else:
                    end += 1
            blank(content_start, end)
            position = min(end + len(delimiter), length)
        else:
            position += 1
    return "".join(result)
