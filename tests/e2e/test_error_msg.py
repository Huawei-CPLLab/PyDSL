from pydsl.compiler import CompilationError
from pydsl.frontend import compile
from helper import run

import os
import inspect


def test_error_msg():
    try:

        @compile()
        def f():
            return a

    except CompilationError as e:
        current_line = inspect.currentframe().f_lineno
        # the number 2 is the relative position
        # between inspect.currentframe() and the line `return a`
        error_line = current_line - 3

        current_file_path = os.path.abspath(__file__)

        expected_msg = f'File "{current_file_path}", line {error_line}'

        assert e.programmer_message().startswith(expected_msg), (
            f"expect error message:\n"
            f"{expected_msg}\n"
            f"got\n"
            f"{e.programmer_message()}"
        )
    else:
        assert False, f"expected CompilationError, but no error was raised"


if __name__ == "__main__":
    run(test_error_msg)
