"""
Decorator for tail recursive calls.


How to use:

- Decorate your function with `@recursive`
- Return a `TailCall` instance if you want to do a tail call
  of the function being decorated
- Return anything else will terminate the function


Note:

This package doesn't support non-tail recursive calls.
Sometimes you can add the state to function arguments to make it tail-recursive,
other times this cannot be easily done, in that case you still need to use loops.


Example:

```
# return the cumulative sum of a list
from typing import List

@recursive
def cum_sum(xs: List[int], sums: List[int], cur_sum: int):
    if not xs:
        return sums
    else:
        h = xs[0]
        t = xs[1:]
        cur_sum += h
        sums.append(cur_sum)
        return TailCall(t, sums, cur_sum)

assert cum_sum([2, 3, 5, 7, 11], [], 0) == [2, 5, 10, 17, 28]
```
"""


class TailCall:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

def recursive(f):

    def wrapped(*args, **kwargs):
        args = args
        kwargs = kwargs
        while True:
            res = f(*args, **kwargs)
            if isinstance(res, TailCall):
                args = res.args
                kwargs = res.kwargs
            else:
                return res
    return wrapped
