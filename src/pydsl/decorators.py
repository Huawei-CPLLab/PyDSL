from abc import ABC, abstractmethod


class Decorator(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self):
        """
        This function described the transformation
        the decorator applies to the function it is
        passed as an argument

        The syntax is

        ```
        @compile()
        @Decorator
        def my_func(...) -> ...:
            ...
        ```

        The decorator is applied after the function is
        compiled to MLIR.
        """
        ...
