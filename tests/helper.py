from contextlib import contextmanager
import gc
import sys
import math
import numpy as np
import re

from pydsl.compiler import CompilationError


@contextmanager
def failed_from(error_type: type[Exception]):
    """
    Same purpose as pytest.raises, was added when we were using llvm-lit
    instead of PyTest. Keeping this since its error messages are a bit more
    descriptive than pytest.raises.
    """
    try:
        yield
    except Exception as e:
        assert isinstance(
            e, error_type
        ), f"expected {error_type}, but instead got {type(e)}"
    else:
        assert False, f"expected {error_type}, but no error was raised"


@contextmanager
def compilation_failed_from(error_type: type[Exception]):
    try:
        yield
    except Exception as e:
        assert isinstance(
            e, CompilationError
        ), f"expected CompilationError, but instead got {type(e)}"

        assert isinstance(
            e.exception, error_type
        ), f"CompilationError is caused by {e.exception}, but should be {error_type}"
    else:
        assert False, f"expected CompilationError caused by {error_type}, but no error was raised"


def multi_arange(shape: tuple[int], dtype: type) -> np.ndarray:
    """
    Returns an ndarray with the given shape and values filled in
    as 0, 1, 2, ...
    """
    return np.arange(math.prod(shape), dtype=dtype).reshape(shape)


# Log everything to stderr and flush so that we have a unified stream to match
# errors/info emitted by MLIR to stderr.
def log(*args):
    print(*args, file=sys.stderr)
    sys.stderr.flush()


# Implemented for llvm-lit, kept because this is nice if you want to test an
# individual file instead of using hatch test everything.
def run(f):
    log("\nTEST:", f.__name__)
    f()
    log("SUCCESS:", f.__name__)
    gc.collect()


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
