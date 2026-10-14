import os
from dotenv import load_dotenv
load_dotenv(".env")
val = os.getenv("CHANGE_HEAD_WEIGHTS", "NOT SET")
print(f"CHANGE_HEAD_WEIGHTS = '{val}'")
print(f"File exists: {os.path.exists(val)}")
