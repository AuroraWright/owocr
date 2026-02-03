import multiprocessing
import sys
import os

def main():
    multiprocessing.set_start_method('spawn')
    if sys.platform == 'darwin':
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    try:
        from .run import run
    except ImportError:
        from owocr.run import run
    run()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    import pip_system_certs.wrapt_requests
    pip_system_certs.wrapt_requests.inject_truststore()
    main()
