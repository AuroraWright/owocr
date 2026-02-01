import tkinter as tk
from tkinter import PhotoImage, ttk
import multiprocessing
import queue
import sys
import importlib.resources

class LogViewer:
    def __init__(self, root, log_queue):
        self.root = root
        self.log_queue = log_queue

        self._setup_window()
        self._initialize_styles()
        self._create_widgets()

        self.after_id = None
        self._poll_queue()

    def _setup_window(self):
        self.root.title('owocr Log Viewer')
        icon_path = importlib.resources.files(__name__).joinpath('data', 'icon.png')
        icon = PhotoImage(file=icon_path)
        self.root.iconphoto(True, icon)

        width, height = 900, 600
        window_scale = 1.0
        if sys.platform == 'win32':
            hwnd = self.root.winfo_id()
            dpi = ctypes.windll.user32.GetDpiForWindow(hwnd)
            window_scale = dpi / 96.0

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        width = int(width * window_scale)
        height = int(height * window_scale)
        x = screen_width // 2 - width // 2
        y = screen_height // 2 - height // 2

        self.root.geometry(f'{width}x{height}+{x}+{y}')
        self.root.resizable(False, False)

    def _initialize_styles(self):
        style = ttk.Style()

        if sys.platform == 'linux':
            style.theme_use('alt')

    def _create_widgets(self):
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill='x', padx=10, pady=10)

        self.auto_scroll = tk.BooleanVar(value=True)
        auto_scroll_cb = ttk.Checkbutton(control_frame, text='Auto-scroll', variable=self.auto_scroll)
        auto_scroll_cb.pack(side='left', padx=5)

        self.status_label = ttk.Label(control_frame, text='Running', foreground='green')
        self.status_label.pack(side='left', padx=10)

        text_frame = ttk.Frame(self.root)
        text_frame.pack(fill='both', expand=True, padx=10, pady=(0,20))

        self.text_area = tk.Text(
            text_frame,
            wrap='word',
            font=('Courier New', 10),
            bg='white',
            fg='black',
            state='disabled',
            borderwidth=1
        )

        scrollbar = ttk.Scrollbar(text_frame, orient='vertical', command=self.text_area.yview)
        self.text_area.configure(yscrollcommand=scrollbar.set)

        self.text_area.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        self.text_area.tag_config('TIMESTAMP', foreground='blue')
        self.text_area.tag_config('DEBUG', foreground='purple')
        self.text_area.tag_config('INFO', foreground='black')
        self.text_area.tag_config('WARNING', foreground='orange')
        self.text_area.tag_config('ERROR', foreground='red')

    def _poll_queue(self):
        terminated = False
        if not self.log_queue.empty():
            self.text_area.configure(state='normal')
            try:
                while True:
                    record = self.log_queue.get_nowait()
                    self._process_record(record)
                    if record['message'] == 'Terminated!':
                        terminated = True
            except queue.Empty:
                self.text_area.configure(state='disabled')

        if not terminated:
            self.after_id = self.root.after(100, self._poll_queue)

    def _process_record(self, record):
        time_str = record['time'].strftime('%H:%M:%S')
        message = record['message']
        formatted = f'{time_str} | {message}'

        current_pos = self.text_area.index('end-1c')

        self.text_area.insert(tk.END, formatted + '\n')

        timestamp_end = f'{current_pos}+8c'
        self.text_area.tag_add('TIMESTAMP', current_pos, timestamp_end)

        level_name = record['level'].name
        line_end = self.text_area.index('end-2c')
        self.text_area.tag_add(level_name, timestamp_end, line_end)

        if self.auto_scroll.get():
            self.text_area.see(tk.END)

        if record['level'].name == 'ERROR':
            self.status_label.config(text='Stopped', foreground='red')


def main(log_queue):
    root = tk.Tk()
    app = LogViewer(root, log_queue)

    def on_closing():
        if app.after_id:
            root.after_cancel(app.after_id)
        root.destroy()

    root.protocol('WM_DELETE_WINDOW', on_closing)
    root.mainloop()
    root.update()
