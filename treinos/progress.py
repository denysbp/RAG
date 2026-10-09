from tqdm import tqdm
import time


def progress(number: int) -> None:
    for i in tqdm(range(0, number), desc="Progress"):
        time.sleep(0.01)


if __name__ == "__main__":
    progress(1000)
