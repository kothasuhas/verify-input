import sys

from abcrown import ABCROWN


if __name__ == '__main__':
    abcrown = ABCROWN(args=sys.argv[1:] + ['--return_optimized_model'])
    model = abcrown.main()

    print(model.net['/18'].lower)
    print(model.net['/18'].upper)
