from dataclasses import dataclass
from timeit import Timer
from typing import Any, Callable, Self
from copy import deepcopy
import inspect
from warnings import warn

from pydsl.frontend import CompiledFunction, compile as pydslcompile

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
ENDC = "\033[0m"


#####################################################################
# Autotune Configuration Logic
#####################################################################


@dataclass
class Config:
    """
    This class represents a possible configuration
    that autotune may test
    """

    # environment arguments
    env: dict[str, Any]

    # testing data arguments
    args: list[Any]

    # compilation arguments
    # will be spat out like pydslcompile(context, **settings)
    settings: dict[str, Any]
    # context: Optional[dict[str, Any]] = None, user can just pass this in with the decorators

    def __init__(self, env=None, args=None, settings=None):
        self.env = env if env else dict()
        self.args = args
        self.settings = settings if settings else dict()

    def __repr__(self):
        return f"Config(\n env: {CYAN}{self.env}{ENDC},\n settings:{CYAN}{self.settings}{ENDC}\n)"

    def union(cfg1: Self, cfg2: Self) -> Self:
        # Technically, cfg1 should be called, "self", but
        # its fine...
        # use this function like this
        # cfg_new = cfg1.union(cfg2)

        def helper(d1, d2, msg):
            # a quick helper function that checks to see
            # that for all keys common to both dictionaries,
            # the corresponding values are equal
            for k in set(d1).intersection(set(d2)):
                if d1[k] != d2[k]:
                    raise ValueError(
                        f"{msg}, key={k}, value 1 = {d1[k]}, value 2 = {d2[k]}"
                    )

        helper(
            cfg1.env,
            cfg2.env,
            "Two configurations attempted to set conflicting environment variables.",
        )

        new_env = cfg1.env | cfg2.env

        if (
            cfg1.args is not None
            and cfg2.args is not None
            and cfg1.args != cfg2.args
        ):
            raise ValueError(
                "Two configurations attempted to set conflicting testing data.\n {cfg1}\n {cfg2}\n"
            )

        new_args = cfg2.args if cfg2.args is not None else cfg1.args

        helper(
            cfg1.settings,
            cfg2.settings,
            "Two configurations attempted to set conflicting compilation settings.",
        )

        new_settings = cfg1.settings | cfg2.settings

        return Config(new_env, new_args, new_settings)


@dataclass
class AutotuneConfigurations:
    """
    This class holds a list of Configs, along with providing various builder
    methods to help with designing custom autotune sequences.
    """

    configs: list[Config]

    def concat(self, other: Self) -> Self:
        return AutotuneConfigurations(self.configs + other.configs)

    def cartesian(self, other: Self) -> Self:
        return AutotuneConfigurations([
            cfg1.union(cfg2) for cfg1 in self.configs for cfg2 in other.configs
        ])

    def zip(self, other: Self) -> Self:
        return AutotuneConfigurations([
            cfg1.union(cfg2)
            for cfg1, cfg2 in zip(self.configs, other.configs, strict=True)
        ])

    def __add__(self, other: Self) -> Self:
        return self.concat(other)

    def __mul__(self, other: Self) -> Self:
        return self.cartesian(other)

    def __xor__(self, other: Self) -> Self:
        return self.zip(other)


# These are a bunch of methods to help build AutotuneConfigurations


def Default(length: int = 1):
    return AutotuneConfigurations([Config() for _ in range(length)])


def Var(name: str, vals: list[Any]):
    return AutotuneConfigurations([Config(env={name: val}) for val in vals])


class Setting:
    def __init__(self):
        raise ValueError(
            "This class just holds a bunch of methods, don't construct an instance of it."
        )

    @classmethod
    def set(cls, name, vals):
        return AutotuneConfigurations([
            Config(settings={name: val}) for val in vals
        ])

    @classmethod
    def transform_seq(cls, vals):
        return AutotuneConfigurations([
            Config(settings={"transform_seq": val}) for val in vals
        ])

    @classmethod
    def target_class(cls, vals):
        return AutotuneConfigurations([
            Config(settings={"target_class": val}) for val in vals
        ])

    # ... add performance relevent settings here as needed


def TestingData(vals: list[Any]):
    return AutotuneConfigurations([Config(args=val) for val in vals])


#####################################################################
# Autotuning
#####################################################################


def autotune(
    at_cfg: AutotuneConfigurations, context=None, verbose=True
) -> Callable[[Callable], CompiledFunction]:
    """
    This function takes a series of autotune configurations, compiles the given
    function using each one, and returns the CompiledFunction with the best performance

    This is a function that returns a function which can be used as a decorator

    autotune(cfg) # is a function

    @autotune(cfg)
    def my_func(...):
        ...

    is a compiled function
    """

    if context is None:
        # this code is stolen from frontend.py to avoid passing None
        # to pydsl.frontend.compile. The reason we don't pass
        # None is that it would grab the stack frame of this
        # function which is not what we want.

        f_back = inspect.currentframe().f_back

        context = dict(
            dict(f_back.f_builtins, **f_back.f_globals), **f_back.f_locals
        )

    def payload(f: Callable) -> CompiledFunction:
        candidate_functions: list[tuple[CompiledFunction, Config, float]] = []

        for cfg in at_cfg.configs:
            if verbose:
                print("=" * 100)
                print(f"Testing the following configuration: {cfg}")

            comp_func = pydslcompile(context | cfg.env, **cfg.settings)(f)
            args = cfg.args
            if args is None:
                warn("No testing data provided, defaulting to []")
                args = []
            dc_args = deepcopy(args)
            t = Timer(lambda: comp_func(*dc_args)).timeit(1)

            candidate_functions.append((comp_func, cfg, t))

            if verbose:
                print(f"Time achieved: {YELLOW}{t}{ENDC} seconds")

        best_func, best_cfg, best_time = min(
            candidate_functions, key=lambda x: x[2]
        )

        if verbose:
            print(
                "=" * 100
                + f"\nBest Time: {GREEN}{best_time}{ENDC} achieved by Best Config: {best_cfg}\n"
                + "=" * 100
            )

        return best_func

    return payload
