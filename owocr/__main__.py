import fire
import inspect
from owocr.run import run, init_config

def main():
    init_config()

    from owocr.run import config
    cli_args = inspect.getfullargspec(run)[0]
    defaults = []

    index = 0
    for arg in cli_args:
        defaults.append(config.get_general(arg))
        index += 1

    run.__defaults__ = tuple(defaults)

    fire.Fire(run)


if __name__ == '__main__':
    main()
