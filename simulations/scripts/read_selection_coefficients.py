from pathlib import Path
import pandas as pd

class Reader:
    def __init__(self, folder: str, mode="local"):
        if mode == "local":
            root_folder =  "../data/selection_coefficients"
        elif mode == "cluster":
            root_folder = "./selection_coefficients"
        else:
            raise ValueError(f"Unknown mode: {mode}")
        files = Path(f"{root_folder}/{folder}").glob("*")
        parts = []
        for file in files:
            if file.name.endswith("json"):
                continue
            parts.append(pd.read_csv(file, sep='\t'))
        self.df = pd.concat(parts, ignore_index=True)


reader = Reader("linear_da_lower_cost", mode="cluster")
reader.df.to_csv("selection_coefficients_df.tsv", sep="\t", index=False)
