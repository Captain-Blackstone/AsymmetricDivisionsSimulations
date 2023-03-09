from pathlib import Path
lines = []
for file in Path("equilibria/").glob("*"):
    params = list(map(float, file.stem.split("_")))
    with file.open("r") as fl:
        popsize = float(fl.readlines()[2].strip())
        params.append(popsize)
    lines.append("_".join(list(map(str, params))))
with open("parameter_search.tsv", "w") as fl:
    fl.write("asymmetry\trepair\tA\tB\tC\tD\tE\tF\tN\n")
    fl.write("\n".join(lines))
