import multiprocessing
import queue
import mss
from loguru import logger
from PIL import Image
import sys
try:
    from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
except ImportError:
    pass

try:
    from PIL import ImageTk
    import tkinter as tk
    selector_available = True
except:
    selector_available = False


class ScreenSelector:
    def __init__(self, result_queue, command_queue):
        self.sct = mss.mss()
        self.monitors = self.sct.monitors[1:]
        self.root = None
        self.result_queue = result_queue
        self.command_queue = command_queue
        self.mac_init_done = False

    def on_select(self, monitor, coordinates):
        self.result_queue.put({'monitor': monitor, 'coordinates': coordinates})
        if self.root:
            self.root.destroy()

    def _setup_selection_canvas(self, canvas, img_tk, scale_x=1, scale_y=1, monitor=None):
        canvas.pack(fill=tk.BOTH, expand=True)
        canvas.image = img_tk
        canvas.create_image(0, 0, image=img_tk, anchor=tk.NW)

        start_x, start_y, rect = None, None, None

        def on_click(event):
            nonlocal start_x, start_y, rect
            start_x, start_y = event.x, event.y
            rect = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline='red')

        def on_drag(event):
            nonlocal rect, start_x, start_y
            if rect:
                canvas.coords(rect, start_x, start_y, event.x, event.y)

        def on_release(event):
            nonlocal start_x, start_y
            if start_x is None or start_y is None:
                return

            end_x, end_y = event.x, event.y

            x1 = min(start_x, end_x) 
            y1 = min(start_y, end_y) 
            x2 = max(start_x, end_x) 
            y2 = max(start_y, end_y)

            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            self.on_select(monitor, (x1, y1, x2 - x1, y2 - y1))

        def reset_selection(event):
            nonlocal start_x, start_y, rect
            if rect:
                canvas.delete(rect)
                rect = None
            start_x = None
            start_y = None

        canvas.bind('<ButtonPress-1>', on_click)
        canvas.bind('<B1-Motion>', on_drag)
        canvas.bind('<ButtonRelease-1>', on_release)
        canvas.bind('<Leave>', reset_selection)

    def _create_selection_window(self, img, geometry, scale_x=1, scale_y=1, monitor=None):
        window = tk.Toplevel(self.root)
        window.geometry(geometry)
        window.overrideredirect(1)
        window.attributes('-topmost', 1)

        img_tk = ImageTk.PhotoImage(img)
        canvas = tk.Canvas(window, cursor='cross', highlightthickness=0)

        self._setup_selection_canvas(canvas, img_tk, scale_x, scale_y, monitor)

    def create_window_from_image(self, img):
        original_width, original_height = img.size
        display_monitor = None

        for monitor in self.monitors:
            if (monitor['width'] >= original_width and 
                monitor['height'] >= original_height):
                display_monitor = monitor
                break

        if not display_monitor:
            display_monitor = self.monitors[0]

        window_width = min(original_width, display_monitor['width'])
        window_height = min(original_height, display_monitor['height'])
        left = display_monitor['left'] + (display_monitor['width'] - window_width) // 2
        top = display_monitor['top'] + (display_monitor['height'] - window_height) // 2

        geometry = f"{window_width}x{window_height}+{left}+{top}"

        if img.width > window_width or img.height > window_height:
            img = img.resize((window_width, window_height), Image.Resampling.LANCZOS)
            scale_x = original_width / window_width
            scale_y = original_height / window_height
        else:
            scale_x = 1
            scale_y = 1

        self._create_selection_window(img, geometry, scale_x, scale_y, None)

    def create_window(self, monitor):
        screenshot = self.sct.grab(monitor)
        img = Image.frombytes('RGB', screenshot.size, screenshot.rgb)
        original_width, original_height = img.size

        geometry = f"{monitor['width']}x{monitor['height']}+{monitor['left']}+{monitor['top']}"

        if img.width != monitor['width']:
            img = img.resize((monitor['width'], monitor['height']), Image.Resampling.LANCZOS)
            scale_x = original_width / monitor['width']
            scale_y = original_height / monitor['height']
        else:
            scale_x = 1
            scale_y = 1

        self._create_selection_window(img, geometry, scale_x, scale_y, monitor)

    def start(self):
        while True:
            image = self.command_queue.get()

            if image == False:
                break
            if image == True:
                self.on_select(None, None)
                continue

            self.root = tk.Tk()

            if not self.mac_init_done and sys.platform == 'darwin':
                app = NSApplication.sharedApplication()
                app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
                self.mac_init_done = True

            self.root.withdraw()

            if image:
                self.create_window_from_image(image)
            else:
                for monitor in self.monitors:
                    self.create_window(monitor)

            self.root.mainloop()
            self.root.update()
            self.root = None


def run_screen_selector(result_queue, command_queue):
    selector = ScreenSelector(result_queue, command_queue)
    selector.start()

selector_process = None
result_queue = None
command_queue = None

def get_screen_selection(pil_image, permanent_process):
    global selector_process, result_queue, command_queue

    if not selector_available:
        logger.error('tkinter or PIL with tkinter support are not installed, unable to open picker')
        sys.exit(1)

    if selector_process is None or not selector_process.is_alive():
        result_queue = multiprocessing.Queue()
        command_queue = multiprocessing.Queue()
        selector_process = multiprocessing.Process(target=run_screen_selector, args=(result_queue, command_queue))
        selector_process.daemon = True
        selector_process.start()

    command_queue.put(pil_image)

    result = False
    while (not result) and selector_process.is_alive():
        try:
            result = result_queue.get(timeout=0.1)
        except:
            continue
    if not permanent_process:
        terminate_selector_if_running()
    return result

def terminate_selector_if_running():
    if selector_process and selector_process.is_alive():
        command_queue.put(False)
        selector_process.join()
