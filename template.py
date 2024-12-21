import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
    "src/__init__.py",
    "src/prompt.py",
    "src/helper.py",
    "src/llm.py",
    "setup.py",
    "app.py",
    "store_index.py",
    "static",
    "templates/chat.html",


]

for file in list_of_files:
    file_path = Path(file)
    file_dir, file_name = os.path.split(file_path)
    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"Creating directory: {file_dir} for file: {file_name}")
    
    if (not os.path.exists(file_path)) or (os.path.getsize(file_name) == 0):
        with open(file_path, "w") as f:
            pass
            logging.info(f"Creating empty file: {file_path}")

    else:
        logging.info(f"{file_name} already exists")