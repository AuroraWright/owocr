from src.run import aaa
import multiprocessing

def main():
    aaa()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
