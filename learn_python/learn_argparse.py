#https://docs.python.org/3/howto/argparse.html

import argparse
parser = argparse.ArgumentParser()
#parser.parse_args()

#run following
#python learn_argparse.py --help
#python learn_argparse.py --verbose
#python learn_argparse.py foo


parser.add_argument("echo", help="echo the string you use here")
parser.add_argument("-v","--verbosity", help="increase output verbosity",
                    action="store_true")
args = parser.parse_args()
print(f"echo: {args.echo}")
print(f"verbose: {args.verbosity}")





print(None)