"""Code entities without interpreting user-supplied Markdown or HTML."""
import re

TOKENS = re.compile(
    r"(?<![\w])(?:sync:[0-9a-f]{12,32}|[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}"
    r"|#[0-9a-f]{8,32}|-?[1-9][0-9]{4,15}|/(?:spb_sync|sync|sudo|unsudo|purge|evidence|help|start|join|ignore|queue))(?![\w])",
    re.IGNORECASE,
)


def code_spans(text):
    urls = [m.span() for m in re.finditer(r'https?://\S+', text)]
    result = []
    for match in TOKENS.finditer(text):
        if any(start < match.end() and match.start() < end for start, end in urls):
            continue
        offset = len(text[:match.start()].encode('utf-16-le')) // 2
        length = len(match[0].encode('utf-16-le')) // 2
        result.append((offset, length))
    return result
