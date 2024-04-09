import os
import zipfile


def unzip_file(file_path, output_dir):
    # create output dir
    os.makedirs(output_dir, exist_ok=True)

    # unzip file
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    print("Unzip to :")
    files = os.listdir(output_dir)
    for file in files:
        print(os.path.join(output_dir, file))

# example
# file_path = "example.zip"
# output_dir = "output_folder"
# unzip_file(file_path, output_dir)
