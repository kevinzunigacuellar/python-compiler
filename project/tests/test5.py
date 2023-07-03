def add(a: int, b: int, c: int, d: int, e: int, f: int, g: int) -> int:
    return a + b + c + d + e + f + g


x = add(1, 2, 3, 4, 5, 6, 7) + 1
y = add(1, 2, 3, 4, 5, 6, 7) - 1

print(x == y)
