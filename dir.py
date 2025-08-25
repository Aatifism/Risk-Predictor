import os

def display_directory_structure(start_path, indent=""):
    try:
        files = os.listdir(start_path)
    except PermissionError:
        print(indent + "[Permission Denied]")
        return

    for i, file in enumerate(files):
        path = os.path.join(start_path, file)
        is_last = (i == len(files) - 1)
        connector = "└── " if is_last else "├── "
        print(indent + connector + file)

        if os.path.isdir(path):
            extension = "    " if is_last else "│   "
            display_directory_structure(path, indent + extension)

# Change this path to your desired directory
directory_path = "."
display_directory_structure(directory_path)
