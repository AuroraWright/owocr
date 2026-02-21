import multiprocessing
import sys
import os

def main():
    if not getattr(sys, 'frozen', False):
        original_stderr_fd = os.dup(sys.stderr.fileno())
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stderr.fileno())
        sys.stderr = os.fdopen(original_stderr_fd, 'w')
    multiprocessing.set_start_method('spawn')
    if sys.platform == 'darwin':
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    from owocr.run import run
    run()

if __name__ == '__main__':
    if getattr(sys, 'frozen', False):
        import pip_system_certs.wrapt_requests
        multiprocessing.freeze_support()
        pip_system_certs.wrapt_requests.inject_truststore()
    main()
