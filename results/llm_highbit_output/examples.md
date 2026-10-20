# High-bit output examples

## Pythia-160M

**garbled** — cont_02, layer E, bit 30, char_sim 0.585

- prompt: `The quick brown fox jumps over the lazy`
- clean: `dog, and the quick brown fox jumps over the lazy dog.\n\nThe quick brown fox jumps over the lazy dog, and the quick brown fox jumps over the lazy dog.\n\nThe quick brown fox jumps over the lazy dog,`
- fault: `The quick brown fox jumps over the quick brown fox jumps over the quick brown fox jumps over the quick brown fox jumps over the quick brown fox jumps over the quick brown fox jumps over the quick brown fox jumps over the`

**wrong** — code_01, layer E, bit 30, char_sim 0.743

- prompt: `def add(a, b):\n    return a + b\n\ndef mul(a, b):\n    return`
- clean: `a * b\n\ndef mul_2(a, b):\n    return a * b\n\ndef mul_3(a, b):\n    return a * b\n\ndef mul_4(a, b):`
- fault: `return a + b\n\ndef mul(a, b):\n    return a + b\n\ndef mul(a, b):\n    return a + b\n\ndef mul(a, b):\n    return`

**preserved** — code_01, layer E, bit 28, char_sim 1.000

- prompt: `def add(a, b):\n    return a + b\n\ndef mul(a, b):\n    return`
- clean: `a * b\n\ndef mul_2(a, b):\n    return a * b\n\ndef mul_3(a, b):\n    return a * b\n\ndef mul_4(a, b):`
- fault: `a * b\n\ndef mul_2(a, b):\n    return a * b\n\ndef mul_3(a, b):\n    return a * b\n\ndef mul_4(a, b):`

## Pythia-410M

**garbled** — code_04, layer L, bit 28, char_sim 0.148

- prompt: `In Python, 1 == 1 evaluates to`
- clean: `True.\n\nA:\n\nPython's == operator is a bit of a pain to use.  It's not as efficient as the == operator in C++.  It's also not as fast as the == operator in C`
- fault: `(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1`

**wrong** — code_01, layer E, bit 30, char_sim 0.657

- prompt: `def add(a, b):\n    return a + b\n\ndef mul(a, b):\n    return`
- clean: `a * b\n\ndef div(a, b):\n    return a / b\n\ndef pow(a, b):\n    return a ** b\n\ndef pow2(a, b):\n    return a **`
- fault: `, c = a, b\n    return c * c\n\ndef div(a, b):\n    return a / b\n\ndef pow(a, b):\n    return a ** b\n\ndef pow2(`

**preserved** — code_01, layer E, bit 28, char_sim 1.000

- prompt: `def add(a, b):\n    return a + b\n\ndef mul(a, b):\n    return`
- clean: `a * b\n\ndef div(a, b):\n    return a / b\n\ndef pow(a, b):\n    return a ** b\n\ndef pow2(a, b):\n    return a **`
- fault: `a * b\n\ndef div(a, b):\n    return a / b\n\ndef pow(a, b):\n    return a ** b\n\ndef pow2(a, b):\n    return a **`

## Pythia-1B

**garbled** — cont_01, layer L, bit 28, char_sim 0.086

- prompt: `Roses are red, violets are`
- clean: `blue, and the sun is yellow.\n\nThe sun is yellow.\n\nThe sun is yellow.\n\nThe sun is yellow.\n\nThe sun is yellow.\n\nThe sun is yellow.\n\nThe sun is`
- fault: `-\n\n-\n\n-\n\n-\n\n-\n\n-\n\n-\n\n-\n\n-\n\n-\n\n-\n\n-\n\n-\n\n-\n\n-\n\n-`

**wrong** — arith_01, layer E, bit 28, char_sim 0.968

- prompt: `1 + 1 = 2. 2 + 2 = 4. 2 + 3 =`
- clean: `5. 2 + 4 = 7. 2 + 5 = 10. 2 + 6 = 13. 2 + 7 = 16. 2 + 8 = 19. 2 + 9 = 22. 2 + 10 = 25. 2 + 11 =`
- fault: `6. 2 + 4 = 8. 2 + 5 = 11. 2 + 6 = 13. 2 + 7 = 16. 2 + 8 = 19. 2 + 9 = 22. 2 + 10 = 25. 2 + 11 =`

**preserved** — arith_01, layer L, bit 30, char_sim 1.000

- prompt: `1 + 1 = 2. 2 + 2 = 4. 2 + 3 =`
- clean: `5. 2 + 4 = 7. 2 + 5 = 10. 2 + 6 = 13. 2 + 7 = 16. 2 + 8 = 19. 2 + 9 = 22. 2 + 10 = 25. 2 + 11 =`
- fault: `5. 2 + 4 = 7. 2 + 5 = 10. 2 + 6 = 13. 2 + 7 = 16. 2 + 8 = 19. 2 + 9 = 22. 2 + 10 = 25. 2 + 11 =`

## Qwen2.5-0.5B

**garbled** — arith_02, layer L, bit 28, char_sim 0.160

- prompt: `3 + 4 = 7. 4 + 5 = 9. 5 + 6 =`
- clean: `11. 6 + 7 = 13. 7 + 8 = 15. 8 + 9 = 17. 9 + 10 = 19. 10`
- fault: `thói quen, thói quen, thói quen, thói quen, thói quen, thói quen, thói quen, thói quen, thói quen, thói quen, thói quen, thói quen,`

**wrong** — arith_01, layer L, bit 28, char_sim 0.719

- prompt: `1 + 1 = 2. 2 + 2 = 4. 2 + 3 =`
- clean: `5. 2 + 4 = 6. 2 + 5 = 7. 2 + 6 = 8. 2 + 7 = 9. 2 + 8 = 10`
- fault: `thói quen, 2 + 4 = 6. 2 + 5 = 8. 2 + 6 = 10. 2 + 7 = 12. 2 + 8 =`

**preserved** — arith_01, layer E, bit 28, char_sim 1.000

- prompt: `1 + 1 = 2. 2 + 2 = 4. 2 + 3 =`
- clean: `5. 2 + 4 = 6. 2 + 5 = 7. 2 + 6 = 8. 2 + 7 = 9. 2 + 8 = 10`
- fault: `5. 2 + 4 = 6. 2 + 5 = 7. 2 + 6 = 8. 2 + 7 = 9. 2 + 8 = 10`

## Qwen2.5-1.5B

**garbled** — def_01, layer M, bit 28, char_sim 0.090

- prompt: `Hot is the opposite of cold. Up is the opposite of`
- clean: `down. Left is the opposite of right. But what is the opposite of up? Is it down or left? The answer is down. This is because up is the opposite of down. So, if you are standing up, you are`
- fault: `．\nA. down\nB. up\nC. down\nD. up\n答案:\nA\n\n下列关于细胞结构和功能的叙述，正确的是____\nA. 细胞膜上的载体蛋白和磷脂`

**wrong** — arith_01, layer E, bit 30, char_sim 0.491

- prompt: `1 + 1 = 2. 2 + 2 = 4. 2 + 3 =`
- clean: `5. 2 + 4 = 6. 2 + 5 = 7. 2 + 6 = 8. 2 + 7 = 9. 2 + 8 = 10`
- fault: `! = 8. 3 + 3! = 12. 3 + 4! = 20. 4 + 4! = 30. 4 + 5! = 54`

**preserved** — arith_01, layer E, bit 28, char_sim 1.000

- prompt: `1 + 1 = 2. 2 + 2 = 4. 2 + 3 =`
- clean: `5. 2 + 4 = 6. 2 + 5 = 7. 2 + 6 = 8. 2 + 7 = 9. 2 + 8 = 10`
- fault: `5. 2 + 4 = 6. 2 + 5 = 7. 2 + 6 = 8. 2 + 7 = 9. 2 + 8 = 10`
