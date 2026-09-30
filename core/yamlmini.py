"""Minimal YAML-subset loader (stdlib only).

Supports the subset used by this lab:
  - block mappings with indentation
  - block sequences ("- item")
  - inline sequences "[a, b, c]"
  - scalars: int, float, bool, null, quoted/unquoted strings
  - full-line and trailing "#" comments (outside quotes)

If PyYAML is installed, load() uses it; otherwise this parser is used.
"""


def _strip_comment(line):
    out = []
    quote = None
    for i, ch in enumerate(line):
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        else:
            if ch in "\"'":
                quote = ch
                out.append(ch)
            elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
                break
            else:
                out.append(ch)
    return "".join(out).rstrip()


def _split_inline(s, seps=","):
    parts, buf, depth, quote = [], [], 0, None
    for ch in s:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
            buf.append(ch)
        elif ch in "[{":
            depth += 1
            buf.append(ch)
        elif ch in "]}":
            depth -= 1
            buf.append(ch)
        elif ch in seps and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if "".join(buf).strip() != "":
        parts.append("".join(buf).strip())
    return parts


def _scalar(s):
    s = s.strip()
    if s == "" or s in ("null", "Null", "NULL", "~"):
        return None
    if s in ("true", "True", "yes", "on"):
        return True
    if s in ("false", "False", "no", "off"):
        return False
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    if s[0] == "[" and s[-1] == "]":
        inner = s[1:-1].strip()
        if inner == "":
            return []
        return [_scalar(x) for x in _split_inline(inner)]
    if s[0] == "{" and s[-1] == "}":
        inner = s[1:-1].strip()
        d = {}
        if inner == "":
            return d
        for pair in _split_inline(inner):
            k, _, v = pair.partition(":")
            d[k.strip()] = _scalar(v)
        return d
    try:
        return int(s, 0)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _split_key(s):
    quote = None
    for i, ch in enumerate(s):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == ":":
            return s[:i].strip(), s[i + 1:]
    raise ValueError("expected 'key:' in line: %r" % s)


def loads(text):
    lines = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        s = _strip_comment(raw)
        if not s.strip():
            continue
        indent = len(s) - len(s.lstrip(" "))
        lines.append((indent, s.strip()))
    if not lines:
        return {}

    pos = [0]

    def parse_block(indent):
        result = None
        while pos[0] < len(lines):
            ind, content = lines[pos[0]]
            if ind < indent:
                break
            if ind > indent:
                raise ValueError("unexpected indent: %r" % content)
            if content == "-" or content.startswith("- "):
                if result is None:
                    result = []
                if not isinstance(result, list):
                    raise ValueError("mixed list/map at %r" % content)
                item = content[1:].strip()
                pos[0] += 1
                if item == "":
                    if pos[0] < len(lines) and lines[pos[0]][0] > indent:
                        result.append(parse_block(lines[pos[0]][0]))
                    else:
                        result.append(None)
                else:
                    result.append(_scalar(item))
            else:
                if result is None:
                    result = {}
                if not isinstance(result, dict):
                    raise ValueError("mixed map/list at %r" % content)
                key, val = _split_key(content)
                pos[0] += 1
                if val.strip() == "":
                    if pos[0] < len(lines) and lines[pos[0]][0] > indent:
                        result[key] = parse_block(lines[pos[0]][0])
                    else:
                        result[key] = None
                else:
                    result[key] = _scalar(val)
        return result

    return parse_block(lines[0][0])


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text)
    except ImportError:
        return loads(text)
