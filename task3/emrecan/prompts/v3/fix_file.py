import argparse

parser = argparse.ArgumentParser("")
parser.add_argument("--lang", required=True, type=str)
parser.add_argument("--run", required=True, type=int)
args = parser.parse_args()

filename = args.lang + "_analysis_epochs5_run" + str(args.run) + ".txt"
line_list = []
with open(filename, "r") as f:
    lines = f.readlines()
    for line in lines[:-1]:
        line_list.append(line.replace(":", " "))
        #break

with open(args.lang + '.dev' + str(args.run), 'w') as f:
    for line in line_list:
        f.write(f"{line}")