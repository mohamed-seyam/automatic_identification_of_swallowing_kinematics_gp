import os

def make_dir(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        print(f'{dir} Directory Already Exists!')

