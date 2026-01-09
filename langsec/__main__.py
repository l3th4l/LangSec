"""langsec entry point wrapper"""

import sys

from langsec import attacker


def main():
    attacker.main(sys.argv[1:])


if __name__ == "__main__":
    main()