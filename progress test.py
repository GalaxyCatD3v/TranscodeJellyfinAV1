from tqdm import tqdm
import time

with tqdm(range(2), position=0, desc="Overall") as overall:
    for i in overall:
        with tqdm(range(6), position=1, desc="Current", leave=False) as inner:
            for j in inner:
                time.sleep(0.05)