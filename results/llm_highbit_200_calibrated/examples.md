# High-bit output examples

## Qwen2.5-0.5B

**garbled** — arit_01, layer M, bit 28, char_sim 0.176

- prompt: `1 + 1 = 2. 3 + 3 = 6. 2 + 3 =`
- clean: `5. 4 + 4 = 8. 1 + 4 = 5. 5 + 5 = 10. 2 + 5 = 7. 6 + 6 = 1`
- fault: `thói quen, thói quen, thói quen, thói quen, thói quen, thói quen, thói quen, thói quen, thói quen, thói quen, thói quen, thói quen,`

**wrong** — arit_01, layer L, bit 28, char_sim 0.474

- prompt: `1 + 1 = 2. 3 + 3 = 6. 2 + 3 =`
- clean: `5. 4 + 4 = 8. 1 + 4 = 5. 5 + 5 = 10. 2 + 5 = 7. 6 + 6 = 1`
- fault: `thói quen. 3 + 3 = 6. 2 + 3 = thói quen. 3 + 3 = 6. 2 + 3 = thói quen. 3 + 3 =`

**preserved** — arit_01, layer E, bit 28, char_sim 1.000

- prompt: `1 + 1 = 2. 3 + 3 = 6. 2 + 3 =`
- clean: `5. 4 + 4 = 8. 1 + 4 = 5. 5 + 5 = 10. 2 + 5 = 7. 6 + 6 = 1`
- fault: `5. 4 + 4 = 8. 1 + 4 = 5. 5 + 5 = 10. 2 + 5 = 7. 6 + 6 = 1`

## Qwen2.5-1.5B

**garbled** — cont_07, layer E, bit 31, char_sim 0.286

- prompt: `The quick brown fox jumps over the lazy dog. Roses are red, violets are blue. Happy birthday to`
- clean: `you! The quick brown fox jumps over the lazy dog. Roses are red, violets are blue. Happy birthday to you! The quick brown fox jumps over the lazy dog. Roses are red, violets are blue. Happy`
- fault: `you! I am a cat. I am a dog. I am a cat. I am a dog. I am a cat. I am a dog. I am a cat. I am a dog. I am a cat. I`

**wrong** — arit_01, layer E, bit 30, char_sim 0.614

- prompt: `1 + 1 = 2. 3 + 3 = 6. 2 + 3 =`
- clean: `5. 3 + 5 = 8. 5 + 7 = 12. 7 + 9 = 16. 9 + 11 = 20. 11 +`
- fault: `! = 11. 3 + 4 = 7. 4 + 5 = 9. 5 + 6 = 11. 6 + 7 = 13. 7 +`

**preserved** — arit_01, layer E, bit 28, char_sim 1.000

- prompt: `1 + 1 = 2. 3 + 3 = 6. 2 + 3 =`
- clean: `5. 3 + 5 = 8. 5 + 7 = 12. 7 + 9 = 16. 9 + 11 = 20. 11 +`
- fault: `5. 3 + 5 = 8. 5 + 7 = 12. 7 + 9 = 16. 9 + 11 = 20. 11 +`
