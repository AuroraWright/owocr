import multiprocessing
import sys
import os

def main():
    if not getattr(sys, 'frozen', False):
        original_stderr_fd = os.dup(sys.stderr.fileno())
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stderr.fileno())
        sys.stderr = os.fdopen(original_stderr_fd, 'w')
        if sys.platform == 'win32':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11) # STD_OUTPUT_HANDLE
            mode = ctypes.c_uint()
            kernel32.GetConsoleMode(handle, ctypes.byref(mode))
            kernel32.SetConsoleMode(handle, mode.value | 0x0004) # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    multiprocessing.set_start_method('spawn')
    if sys.platform == 'darwin':
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    from owocr.run import run
    run()

if __name__ == '__main__':
    if getattr(sys, 'frozen', False):
        if sys.stdout is None:
            sys.stdout = open(os.devnull, 'w')
        if sys.stderr is None:
            sys.stderr = open(os.devnull, 'w')
        import pip_system_certs.wrapt_requests
        multiprocessing.freeze_support()
        pip_system_certs.wrapt_requests.inject_truststore()
    main()
