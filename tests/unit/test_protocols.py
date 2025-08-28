from functools import cache

from pydsl.protocols import canonicalize_args

from helper import failed_from, run


def test_canonicalize_args():
    calls = []

    @canonicalize_args
    @cache
    def f(a, /, b, c=2, *, d=3):
        calls.append((a, b, c, d))
        return a, b, c, d

    f(0, 1, 2, d=3)
    f(0, b=1, c=2, d=3)
    f(0, b=1, c=2)
    f(0, b=1)
    f(0, 1)
    f(0, c=2, d=3, b=1)
    f(0, d=3, b=1)

    f(0, 1, d=9)
    f(0, b=1, d=9, c=2)

    assert calls == [(0, 1, 2, 3), (0, 1, 2, 9)]

    # Make sure it actually returns something
    assert f(0, 1, 2) == (0, 1, 2, 3)

    with failed_from(TypeError):
        # a is pos only
        f(a=0, b=1, c=2, d=3)

    with failed_from(TypeError):
        # d is kw only
        f(0, 1, 2, 3)


if __name__ == "__main__":
    run(test_canonicalize_args)
