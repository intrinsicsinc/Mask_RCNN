import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='')
    args = parser.parse_args()
    return args

args = parse_args()

with open(args.csv, 'r') as fp:
    contents = [float(line) for line in fp.read().strip().split()]

print(max(contents))


