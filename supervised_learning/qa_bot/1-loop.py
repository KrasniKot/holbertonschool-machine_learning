#!/usr/bin/env python3
""" Script that takes in input from the user """


def qa_loop():
    """ Takes an input from the user and prints the response """
    LTAKING = ["exit", "quit", "goodbye", "bye"]  # Leave-taking expressions

    leaving = False
    while not leaving:
        question = input("Q: ").lower()
        response = ""

        if question in LTAKING:
            leaving = True
            response = "Goodbye"

        print(f"A: {response}")


if __name__ == "__main__":
    qa_loop()
