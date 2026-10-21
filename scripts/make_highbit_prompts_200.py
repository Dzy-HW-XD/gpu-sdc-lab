#!/usr/bin/env python3
"""Build the 200-prompt high-bit clean-calibration suite.

Each prompt is a short few-shot/cloze continuation with a single checkable
answer.  Few-shot examples are chosen so the target answer is not copied from
the prompt unless noted by the task itself.
"""
import json
import os

OUT = "data/highbit_prompts_200.json"


def entry(id_prefix, i, category, prefix, prompt, expected):
    return {
        "id": "%s_%02d" % (id_prefix, i),
        "category": category,
        "prompt": prefix + prompt,
        "expected": [expected],
    }


def main():
    prompts = []

    capitals = [
        ("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"),
        ("Germany", "Berlin"), ("Spain", "Madrid"), ("Portugal", "Lisbon"),
        ("United Kingdom", "London"), ("Canada", "Ottawa"),
        ("Australia", "Canberra"), ("Russia", "Moscow"), ("China", "Beijing"),
        ("India", "New Delhi"), ("Egypt", "Cairo"), ("Greece", "Athens"),
        ("Turkey", "Ankara"), ("Netherlands", "Amsterdam"),
        ("Sweden", "Stockholm"), ("Norway", "Oslo"), ("Denmark", "Copenhagen"),
        ("Finland", "Helsinki"), ("Poland", "Warsaw"), ("Austria", "Vienna"),
        ("Belgium", "Brussels"), ("Switzerland", "Bern"),
        ("Ireland", "Dublin"), ("New Zealand", "Wellington"),
        ("South Korea", "Seoul"), ("Thailand", "Bangkok"),
        ("Vietnam", "Hanoi"), ("Indonesia", "Jakarta"),
        ("Philippines", "Manila"), ("Malaysia", "Kuala Lumpur"),
        ("Kenya", "Nairobi"), ("Nigeria", "Abuja"),
        ("South Africa", "Pretoria"), ("Morocco", "Rabat"),
        ("Chile", "Santiago"), ("Peru", "Lima"), ("Venezuela", "Caracas"),
        ("Cuba", "Havana"),
    ]
    for i, (country, capital) in enumerate(capitals, 1):
        examples = [c for c in capitals if c[0] != country][:2]
        prefix = "The capital of %s is %s. The capital of %s is %s. " % (
            examples[0][0], examples[0][1], examples[1][0], examples[1][1])
        prompts.append(entry("fact", i, "factual", prefix,
                             "The capital of %s is" % country, capital))

    add = [(2, 3, 5), (4, 5, 9), (7, 8, 15), (6, 9, 15), (12, 7, 19),
           (8, 8, 16), (9, 4, 13), (5, 7, 12), (11, 6, 17), (13, 5, 18)]
    sub = [(9, 4, 5), (12, 5, 7), (15, 6, 9), (20, 8, 12), (18, 9, 9),
           (14, 7, 7), (11, 3, 8), (17, 8, 9), (16, 5, 11), (19, 7, 12)]
    mul = [(3, 4, 12), (5, 6, 30), (7, 8, 56), (9, 3, 27), (6, 7, 42),
           (8, 9, 72), (4, 11, 44), (12, 3, 36), (5, 9, 45), (7, 6, 42)]
    div = [(12, 3, 4), (20, 5, 4), (18, 6, 3), (24, 8, 3), (35, 7, 5),
           (42, 6, 7), (56, 8, 7), (63, 9, 7), (48, 6, 8), (72, 9, 8)]
    for i, (a, b, c) in enumerate(add + sub + mul + div, 1):
        op = "+" if i <= 10 else ("-" if i <= 20 else ("*" if i <= 30 else "/"))
        prompts.append(entry("arith", i, "arithmetic",
                             "1 + 1 = 2. 3 + 3 = 6. ",
                             "%d %s %d =" % (a, op, b), str(c)))

    seq = [
        ("prime", "The first three prime numbers are 2, 3, and", "5"),
        ("prime", "The first four prime numbers are 2, 3, 5, and", "7"),
        ("prime", "The first five prime numbers are 2, 3, 5, 7, and", "11"),
        ("letter", "The first three letters are A, B, and", "C"),
        ("letter", "The first four letters are A, B, C, and", "D"),
        ("letter", "The first five letters are A, B, C, D, and", "E"),
        ("day", "The first three days are Monday, Tuesday, and", "Wednesday"),
        ("day", "The first four days are Monday, Tuesday, Wednesday, and", "Thursday"),
        ("day", "The first five days are Monday, Tuesday, Wednesday, Thursday, and", "Friday"),
        ("month", "The first three months are January, February, and", "March"),
        ("month", "The first four months are January, February, March, and", "April"),
        ("month", "The first five months are January, February, March, April, and", "May"),
        ("square", "1, 4, 9, 16, and", "25"),
        ("square", "1, 4, 9, 16, 25, and", "36"),
        ("square", "1, 4, 9, 16, 25, 36, and", "49"),
        ("even", "2, 4, 6, 8, and", "10"),
        ("even", "2, 4, 6, 8, 10, and", "12"),
        ("even", "2, 4, 6, 8, 10, 12, and", "14"),
        ("odd", "1, 3, 5, 7, and", "9"),
        ("odd", "1, 3, 5, 7, 9, and", "11"),
        ("odd", "1, 3, 5, 7, 9, 11, and", "13"),
        ("count", "Count from 1 to 5: 1, 2, 3, 4,", "5"),
        ("count", "Count from 1 to 6: 1, 2, 3, 4, 5,", "6"),
        ("count", "Count from 1 to 7: 1, 2, 3, 4, 5, 6,", "7"),
        ("mult3", "3, 6, 9, 12, and", "15"),
        ("mult3", "3, 6, 9, 12, 15, and", "18"),
        ("mult3", "3, 6, 9, 12, 15, 18, and", "21"),
        ("mult5", "5, 10, 15, 20, and", "25"),
        ("mult5", "5, 10, 15, 20, 25, and", "30"),
        ("mult5", "5, 10, 15, 20, 25, 30, and", "35"),
    ]
    examples = {
        "prime": "The first three letters are A, B, and C. The first three days are Monday, Tuesday, and Wednesday. ",
        "letter": "The first three prime numbers are 2, 3, and 5. The first three days are Monday, Tuesday, and Wednesday. ",
        "day": "The first three prime numbers are 2, 3, and 5. The first three letters are A, B, and C. ",
        "month": "The first three prime numbers are 2, 3, and 5. The first three letters are A, B, and C. ",
        "square": "The first three letters are A, B, and C. The first three days are Monday, Tuesday, and Wednesday. ",
        "even": "The first three letters are A, B, and C. The first three days are Monday, Tuesday, and Wednesday. ",
        "odd": "The first three letters are A, B, and C. The first three days are Monday, Tuesday, and Wednesday. ",
        "count": "The first three letters are A, B, and C. The first three days are Monday, Tuesday, and Wednesday. ",
        "mult3": "The first three letters are A, B, and C. The first three days are Monday, Tuesday, and Wednesday. ",
        "mult5": "The first three letters are A, B, and C. The first three days are Monday, Tuesday, and Wednesday. ",
    }
    for i, (kind, prompt, expected) in enumerate(seq, 1):
        prompts.append(entry("list", i, "list", examples[kind], prompt, expected))

    code_keywords = [
        ("The Python keyword to define a function is", "def"),
        ("The Python keyword for a loop is", "for"),
        ("The Python keyword for a while loop is", "while"),
        ("The Python keyword for a conditional is", "if"),
        ("The Python keyword for returning a value is", "return"),
        ("The Python keyword to import a module is", "import"),
        ("The Python keyword to define a class is", "class"),
        ("The Python keyword for no value is", "None"),
        ("The Python boolean for true is", "True"),
        ("The Python boolean for false is", "False"),
    ]
    code_outputs = [
        ("print(2 + 3) outputs", "5"),
        ("print(4 * 5) outputs", "20"),
        ("print(10 - 3) outputs", "7"),
        ("print(20 // 4) outputs", "5"),
        ("print(5 % 2) outputs", "1"),
        ("print(2 ** 3) outputs", "8"),
        ("print(len([1, 2, 3])) outputs", "3"),
        ("print(max([2, 5, 1])) outputs", "5"),
        ("print(min([4, 2, 8])) outputs", "2"),
        ("print(sum([1, 2, 3])) outputs", "6"),
        ("print(sorted([3, 1, 2])[0]) outputs", "1"),
        ("print(len('abc')) outputs", "3"),
        ("print(1 == 1) outputs", "True"),
        ("print(1 == 2) outputs", "False"),
        ("print(int('7')) outputs", "7"),
        ("print(str(3)) outputs", "3"),
        ("print(bool(0)) outputs", "False"),
        ("print(bool(1)) outputs", "True"),
        ("print(type(1) is int) outputs", "True"),
        ("print('a' + 'b') outputs", "ab"),
    ]
    for i, (prompt, expected) in enumerate(code_keywords, 1):
        examples = [(p, e) for p, e in code_keywords if e != expected][:2]
        prefix = "%s %s. %s %s. " % (
            examples[0][0], examples[0][1], examples[1][0], examples[1][1])
        prompts.append(entry("code", i, "code", prefix, prompt, expected))
    for j, (prompt, expected) in enumerate(code_outputs, len(code_keywords) + 1):
        prompts.append(entry("code", j, "code",
                             "print('a') outputs a. print('b') outputs b. ",
                             prompt, expected))

    opposites = [
        ("hot", "cold"), ("up", "down"), ("left", "right"),
        ("big", "small"), ("fast", "slow"), ("day", "night"),
        ("open", "closed"), ("young", "old"), ("wet", "dry"),
        ("full", "empty"), ("hard", "soft"), ("light", "dark"),
        ("near", "far"), ("early", "late"), ("tall", "short"),
        ("rich", "poor"), ("strong", "weak"), ("clean", "dirty"),
        ("safe", "dangerous"), ("true", "false"), ("yes", "no"),
        ("white", "black"), ("summer", "winter"), ("first", "last"),
    ]
    for i, (word, opposite) in enumerate(opposites, 1):
        examples = [(w, o) for w, o in opposites if w != word][:2]
        prefix = "%s is the opposite of %s. %s is the opposite of %s. " % (
            examples[0][0], examples[0][1], examples[1][0], examples[1][1])
        prompts.append(entry("def", i, "definition", prefix,
                             "%s is the opposite of" % word, opposite))
    extra_def = [
        ("The sun rises in the", "east"),
        ("The sun sets in the", "west"),
        ("Water freezes at", "0"),
        ("A triangle has", "three"),
        ("A square has", "four"),
        ("A week has", "seven"),
    ]
    for j, (prompt, expected) in enumerate(extra_def, len(opposites) + 1):
        prompts.append(entry("def", j, "definition",
                             "Hot is the opposite of cold. Up is the opposite of down. ",
                             prompt, expected))

    cont = [
        ("The quick brown fox jumps over the lazy", "dog"),
        ("Roses are red, violets are", "blue"),
        ("Twinkle, twinkle, little", "star"),
        ("To be or not to be, that is the", "question"),
        ("Mary had a little", "lamb"),
        ("The cat sat on the", "mat"),
        ("Happy birthday to", "you"),
        ("Row, row, row your", "boat"),
        ("Jack and Jill went up the", "hill"),
        ("Humpty Dumpty sat on a", "wall"),
        ("Baa, baa, black", "sheep"),
        ("Old MacDonald had a", "farm"),
        ("The itsy bitsy", "spider"),
        ("London Bridge is falling", "down"),
        ("Jingle bells, jingle", "bells"),
        ("We wish you a merry", "Christmas"),
        ("I'm a little teapot, short and", "stout"),
        ("Hey diddle diddle, the cat and the", "fiddle"),
        ("Little Miss Muffet sat on a", "tuffet"),
        ("Peter Piper picked a peck of pickled", "peppers"),
        ("She sells sea shells by the sea", "shore"),
        ("The rain in Spain stays mainly in the", "plain"),
        ("A journey of a thousand miles begins with a single", "step"),
        ("All that glitters is not", "gold"),
        ("Actions speak louder than", "words"),
        ("Better late than", "never"),
        ("Practice makes", "perfect"),
        ("Where there's a will there's a", "way"),
        ("The early bird catches the", "worm"),
        ("The grass is always greener on the other", "side"),
    ]
    for i, (prompt, expected) in enumerate(cont, 1):
        examples = [(p, e) for p, e in cont if p != prompt][:2]
        prefix = "%s %s. %s %s. " % (
            examples[0][0], examples[0][1], examples[1][0], examples[1][1])
        prompts.append(entry("cont", i, "continuation", prefix, prompt, expected))

    # fix IDs to category-specific counters
    counts = {}
    for p in prompts:
        cat = p["category"]
        counts[cat] = counts.get(cat, 0) + 1
        p["id"] = "%s_%02d" % (cat[:4], counts[cat])

    assert len(prompts) == 200, len(prompts)
    assert len({p["id"] for p in prompts}) == 200
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=1)
    print("wrote", OUT, len(prompts))
    from collections import Counter
    print(Counter(p["category"] for p in prompts))


if __name__ == "__main__":
    main()
