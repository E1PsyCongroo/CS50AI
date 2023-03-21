from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# Puzzle 0
# A says "I am both a knight and a knave."
Puzzle0_A = And(AKnave, AKnight)
knowledge0 = And(
    # TODO
    Or(AKnave, AKnight),
    Not(And(AKnave, AKnight)),
    Implication(AKnight, Puzzle0_A),
    Implication(AKnave, Not(Puzzle0_A)),
)

# Puzzle 1
# A says "We are both knaves."
# B says nothing.
Puzzle1_A = And(AKnave, BKnave)
knowledge1 = And(
    # TODO
    Not(Biconditional(AKnight, AKnave)),
    Not(Biconditional(BKnight, BKnave)),
    Biconditional(AKnight, Puzzle1_A),
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
Puzzle2_A = Or(And(AKnave, BKnave), And(AKnight, BKnight))
Puzzle2_B = Or(And(AKnave, BKnight), And(AKnight, BKnave))
knowledge2 = And(
    # TODO
    Not(Biconditional(AKnight, AKnave)),
    Not(Biconditional(BKnight, BKnave)),
    Biconditional(AKnight, Puzzle2_A),
    Biconditional(BKnight, Puzzle2_B),
)

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."
Puzzle3_A = Or(Biconditional(AKnight, AKnight), Biconditional(AKnight, AKnave))
Puzzle3_B = And(CKnave, Biconditional(BKnight, Biconditional(AKnight, AKnight)))
Puzzle3_C = AKnight
knowledge3 = And(
    # TODO
    Not(Biconditional(AKnight, AKnave)),
    Not(Biconditional(BKnight, BKnave)),
    Not(Biconditional(CKnight, CKnave)),
    Biconditional(AKnight, Puzzle3_A),
    Biconditional(BKnight, Puzzle3_B),
    Biconditional(CKnight, Puzzle3_C)
)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()
