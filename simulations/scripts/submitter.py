import os
from pathlib import Path

for el in Path("runs").glob("*.sh"):
    os.system(f"qsub {str(el)} -o outsputs/{el.stem}.o -e errors/{el.stem}.e")
