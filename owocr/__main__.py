import multiprocessing

def main():
    multiprocessing.set_start_method('spawn')
    try:
        from .run import run
    except ImportError:
        from owocr.run import run
    run()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
