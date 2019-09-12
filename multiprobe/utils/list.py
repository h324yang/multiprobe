def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def interlacing_repeat(lst, n):
    for x in lst:
        for _ in range(n):
            yield x


def interlacing_expand(lst, n, callback):
    for idx, x in enumerate(lst):
        for _ in range(n):
            yield x
            x = callback(idx, x)
