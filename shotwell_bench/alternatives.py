"""Expand reference notation: [A / B] means either, [A] means optional."""


def _split_options(value):
    parts = []
    depth = 0
    start = 0
    for index, char in enumerate(value):
        if char == '[':
            depth += 1
        elif char == ']':
            depth -= 1
        elif char == '/' and depth == 0:
            parts.append(value[start:index].strip())
            start = index + 1
    parts.append(value[start:].strip())
    return parts


def expand_alternatives(value, limit=64):
    """Enumerate human-authored alternatives without interpreting their semantics."""
    left = value.find('[')
    if left == -1:
        return [' '.join(value.split())]
    depth = 0
    right = None
    for index in range(left, len(value)):
        if value[index] == '[':
            depth += 1
        elif value[index] == ']':
            depth -= 1
            if depth == 0:
                right = index
                break
    if right is None:
        raise ValueError(f'Unclosed reference alternative: {value}')
    before, inside, after = value[:left], value[left + 1:right], value[right + 1:]
    options = _split_options(inside)
    if len(options) == 1:
        options.append('')
    expanded = []
    for option in options:
        for concrete in expand_alternatives(option, limit):
            for tail in expand_alternatives(after, limit):
                separator = ' ' if concrete and tail and after[:1].isspace() else ''
                text = ' '.join((before + concrete + separator + tail).split())
                if text not in expanded:
                    expanded.append(text)
                if len(expanded) > limit:
                    raise ValueError(f'Too many reference alternatives: {value}')
    return expanded
