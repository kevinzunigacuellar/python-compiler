def add(a: int, b: int, c: int, d: int, e: int, f: (int, int)) -> int:
    return a + b + c + d + e + f[0] + f[1]


print(add(1, 2, 3, 4, 5, (6, 7)))
