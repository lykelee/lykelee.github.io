---
title: "Non-recursive Implementation of Depth-first Search (DFS)"
categories:
  - Algorithm
tags:
  - Algorithm
---

Depth-first Search (DFS) Algorithm is a basic algorithm of graph data structures.
In many cases, DFS is implemented by recursive function calls.
The following is a simple Python implementations.

```python
"""
M: A set of marked vertices.
v: The start vertex.
"""
def DFS(M, v):
  M.add(v)
  task_before_entry(v)
  for w in v.neighbors:
    if v not in M:
      DFS(M, v)
  task_after_exit()
```

However, recursive implementations have several problems.
First of all, these are not so efficient in general, because a function call is a quite expensive operation.
Furthermore, each function of call stacks occupy memory.
It sometimes cause a stack overflow.

To overcome these problems, we can consider using non-recursive implementation.
Think of that nested function calls act like a stack.

(In Python, there is no default stack class, but for convenience, I assumed that there is.)

```python
"""
v: The start vertex.
"""
def DFS_nonrecursive(v):
  S = stack()
  M = set()
  S.push(v)
  while len(S) != 0:
    v = S.pop()
    if v in M:
      continue
    M.add(v)
    task_before_entry(v)
    for w in v.neighbors:
      S.push(v)
```

However, the above function cannot handle tasks after exit.
For that, we should use a trick.
Let's think of how it works in recursive implementations.
???

```python
"""
v: The start vertex.
"""
def DFS_nonrecursive(v):
  S = stack()
  M_entry = set()
  M_exit = set()
  S.push((v, False))
  S.push((v, True))
  while len(S) != 0:
    v, entry = S.pop()
    if entry:
      if v in M_entry:
        continue
      M_entry.add(v)
      task_before_entry(v)
      for w in v.neighbors:
        S.push((v, False))
        S.push((v, True))
    else: # exit
      if v in M_exit:
        continue
      M_exit.add(v)
      task_after_entry(v)
```

This non-recursive implementation is less intuitive rather than recursive implementations, but it is more efficient and robust.
Generally, stacks are more efficient than function calls.
Moreover, stack data structures generally use heap memory, which has more capacity than stack memory, so it is more safe in memory lacks.
