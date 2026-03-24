from pathlib import Path

def print_tree(dir_path: Path, prefix: str = ""):
    contents = list(dir_path.iterdir())
    pointers = ["├──"] * (len(contents) - 1) + ["└──"]

    for pointer, path in zip(pointers, contents):
        print(prefix + pointer + " " + path.name)
        if path.is_dir():
            extension = "│   " if pointer == "├──" else "    "
            print_tree(path, prefix + extension)

if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent  
    print(f"{root_dir.name}/")
    print_tree(root_dir)
