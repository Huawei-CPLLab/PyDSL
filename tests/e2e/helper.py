from contextlib import contextmanager
import math
import re

import numpy as np
import pytest

from pydsl.compiler import CompilationError


@contextmanager
def compilation_failed_from(error_type: type[Exception]):
    with pytest.raises(CompilationError) as e:
        yield

    assert isinstance(
        e.value.exception, error_type
    ), f"CompilationError is caused by {e.value.exception}, not {error_type}"


f32info = np.finfo(np.float32)


def f32_isclose(a, b):
    return math.isclose(a, b, rel_tol=f32info.eps)


f32_edges = [
    0.0,
    -0.0,
    math.pi,
    -math.pi,
    1 / math.pi,
    -1 / math.pi,
    f32info.max,
    f32info.min,
    f32info.eps,
    f32info.epsneg,
    math.inf,
    -math.inf,
]


class SequentialTokenConsumer:
    s: str

    def __init__(self, s):
        self.s = s

    def consume_lines(self, match: re.Match):
        """
        Destroy everything in the string from the beginning up to the match,
        the match itself, and everything else on the same line as the match.

        For example, if the match is `"bbb"` in `"aaaabbbcc\\ndddd"`, then the
        string becomes `"dddd"`.
        """
        cutoff = self.s.find("\n", match.end()) + 1

        # cutoff == 0 when no \n is found, since self.s.find(...) returns -1
        if cutoff == 0:
            self.s = ""
        else:
            self.s = self.s[cutoff:]

    def match_single_assignment(self, op_stmt: str) -> str:
        """
        Returns the name of the variable returned by the op_stmt.
        If not present, throws `ValueError`.

        This does not detect for a tuple return.

        E.g. for `%c1 = arith.constant 1 : index`,
        op_stmt = `"arith.constant 1 : index"` returns `"c1"`
        """
        m = re.search(
            r"^\s*%(?P<result>(\d|\w|_)+)\s*=\s*" + re.escape(op_stmt),
            self.s,
            re.MULTILINE,
        )

        if m is None:
            raise ValueError(
                f"no match for operator assignment `{op_stmt}` in:\n{self.s}"
            )
        self.consume_lines(m)

        return m.group("result")

    def match_op(self, op_stmt: str) -> None:
        """
        Check that op_stmt is present as a statement without assign.
        If not present, throws `ValueError`.

        E.g. for `transform.apply_cse to %0 : !transform.any_op`,
        op_stmt = `"transform.apply_cse to %0 : !transform.any_op"` will not
        throw error.
        """
        m = re.search(
            re.escape(op_stmt),
            self.s,
            re.MULTILINE,
        )

        if m is None:
            raise ValueError(
                f"no match for operator `{op_stmt}` in:\n{self.s}"
            )
        self.consume_lines(m)

        return

    def match_trans_seq_arg(self) -> str:
        """
        Returns the name of the single argument of transform sequence.

        E.g. for
        `transform.named_sequence @__transform_main(%arg0: !transform.any_op)`,
        return `arg0`.
        """
        m = re.search(
            r"^\s*transform.named_sequence\s*@.*\s*\(%(?P<arg>(\d|\w|_)+)\s*:.*\)",
            self.s,
            re.MULTILINE,
        )

        if m is None:
            raise ValueError(f"transform sequence not found in:\n{self.s}")
        self.consume_lines(m)

        return m.group("arg")
