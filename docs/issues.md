# Features
- [ ] Add inf, which should automatically support -inf
    - Should be a lazily-evaluated Number to avoid inf32, inf64
- [ ] Given that f is F32 and i is Index, f + i works, but i + f doesn't. Add LHS-implemented op fallback for when RHS cannot cast LHS
    - [ ] This requires other types to know that casting cannot be done with `_try_casting`

# Development features

# Bugs
- [ ] `m[0] += i` results in `AssertionError: subscript with a Store context shouldn't be visited!`
- [ ] Returning a tuple with a single element returns just the element, not that wrapped within a tuple.
    - [ ] Neither does it demand the returning statement to be wrapped in a tuple

# Refactoring


# Not implemented


# Test coverage
- [ ] Add boolean operation tests
- [ ] Add implicit float arith operation tests


# Documentation
- [ ] Improve the examples given in docs/usage.md
- [ ] Change examples that were explicitly defining type casting, especially Index(...)
