import os

def rename_files(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            new_filename = os.path.splitext(filename)[0] + '.sides'
            old_path = os.path.join(directory_path, filename)
            new_path = os.path.join(directory_path, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed '{filename}' to '{new_filename}'")

if __name__ == "__main__":
    directory = '.'
    rename_files(directory)

