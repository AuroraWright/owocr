import fire
import inspect
from owocr.run import run, init_config

def main():
    init_config()

    from owocr.run import config
    fullargspec = inspect.getfullargspec(run)
    old_defaults = fullargspec[0]
    old_default_values = fullargspec[3]
    new_defaults = []

    if config.has_config:
        index = 0
        for argument in old_defaults:
            if config.get_general(argument) == None:
                new_defaults.append(old_default_values[index])
            else:
                new_defaults.append(config.get_general(argument))
            index += 1

        run.__defaults__ = tuple(new_defaults)

    fire.Fire(run)


if __name__ == '__main__':
    main()
