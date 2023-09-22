from process_dset import process_dset

process_dset("./cropped.csv").to_csv("result.csv", index=False)