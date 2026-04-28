import os
import sys
import json
import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Generate rosetta input txt from filtered CSV name.")
    parser.add_argument(
        "--filtered_filename",
        required=True,
        help="Filtered filename stem without .csv, e.g. TSLP_align_pep0_opt_fixed_6959",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    filtered_filename = args.filtered_filename

    df = pd.read_csv(f"/home/lily/SBDD/PocketXMol/TL1A_2026/scripts/{filtered_filename}.csv")
    df = df[["aaseq", "filename_new"]]
    df.columns = ["aaseq", "pdb_ids"]
    df["input_filename"] = df.apply(lambda x: x.aaseq + "_" + x.pdb_ids, axis=1)

    pdb_filespath = "/home/lily/SBDD/PocketXMol/TL1A_2026/results/apo_9/"
    raw_pdb_ids = [x.replace(".pdb", "_complex.pdb") for x in df.input_filename.tolist()]
    pdb_path_lst = [str(os.path.join(pdb_filespath, x)) for x in raw_pdb_ids]

    with open(f"{filtered_filename}.txt", "w") as f:
        for item in pdb_path_lst:
            f.write(str(item) + "#ABC#D" "\n")


if __name__ == "__main__":
    main()