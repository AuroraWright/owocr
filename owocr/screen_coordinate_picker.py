import multiprocessing
import queue
import mss
from loguru import logger
from PIL import Image
from pynput import keyboard
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
        self.after_id = None
        self.result_queue = result_queue
        self.command_queue = command_queue
        self.mac_init_done = False
        self.ctrl_pressed = False
        self.drawing = False
        self.canvases = []
        self.selections = []
        self.keyboard_event_queue = queue.Queue()
        self.start_key_listener()

    def start_key_listener(self):
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release
        )
        self.keyboard_listener.start()

    def on_key_press(self, key):
        if not self.after_id:
            return

        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r, keyboard.Key.cmd_l, keyboard.Key.cmd_r):
            self.ctrl_pressed = True

    def on_key_release(self, key):
        if not self.after_id:
            return

        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r, keyboard.Key.cmd_l, keyboard.Key.cmd_r):
            self.ctrl_pressed = False
            self.keyboard_event_queue.put('return_selections')

        elif key == keyboard.Key.backspace:
            self.keyboard_event_queue.put('clear_selections')

        elif key == keyboard.Key.esc:
            self.keyboard_event_queue.put('return_empty')

    def process_keyboard_events(self):
        if not self.root:
            return

        try:
            while True:
                event_type = self.keyboard_event_queue.get_nowait()
                if event_type == 'return_selections':
                    if self.selections and not self.drawing:
                        self.return_all_selections()
                        return
                elif event_type == 'return_empty':
                    self.return_empty()
                    return
                elif event_type == 'clear_selections':
                    self.clear_all_selections()
        except queue.Empty:
            pass

        if self.root:
            self.after_id = self.root.after(50, self.process_keyboard_events)

    def close_ui(self):
        if self.root:
            if self.after_id:
                try:
                    self.root.after_cancel(self.after_id)
                except:
                    pass
            self.root.destroy()
        self.after_id = None
        self.drawing = False

        self.canvases.clear()
        while not self.keyboard_event_queue.empty():
            self.keyboard_event_queue.get()

    def add_selection(self, monitor, coordinates):
        ctrl_pressed = self.ctrl_pressed

        if coordinates[0] == coordinates[2] or coordinates[1] == coordinates[3]:
            self.clear_all_selections()
            if ctrl_pressed:
                return
            coordinates = None

        self.selections.append({
            'monitor': monitor,
            'coordinates': coordinates
        })

        if not ctrl_pressed:
            self.keyboard_event_queue.put(('return_selections'))
            return

        self.redraw_selections()

    def clear_all_selections(self):
        if self.drawing:
            return

        self.selections.clear()
        self.redraw_selections()

    def return_empty(self):
        self.close_ui()
        self.selections.clear()
        self.result_queue.put(False)

    def return_all_selections(self):
        self.close_ui()

        selections_abs = []
        for selection in self.selections:
            monitor = selection['monitor']
            coordinates = selection['coordinates']
            if monitor and coordinates:
                abs_x1 = monitor['left'] + coordinates[0]
                abs_y1 = monitor['top'] + coordinates[1]
                abs_x2 = monitor['left'] + coordinates[2]
                abs_y2 = monitor['top'] + coordinates[3]
                selections_abs.append({
                    'monitor': monitor,
                    'coordinates': (abs_x1, abs_y1, abs_x2, abs_y2)
                })

        self.selections.clear()
        self.result_queue.put(selections_abs)

    def redraw_selections(self):
        for canvas_info in self.canvases:
            canvas = canvas_info['canvas']
            scale_x = canvas_info['scale_x']
            scale_y = canvas_info['scale_y']
            monitor = canvas_info['monitor']

            items = canvas.find_all()
            for item in items:
                if canvas.gettags(item) and 'selection' in canvas.gettags(item):
                    canvas.delete(item)

            for selection in self.selections:
                if selection['monitor'] == monitor:
                    x1, y1, x2, y2 = selection['coordinates']
                    x1_disp = x1 / scale_x
                    y1_disp = y1 / scale_y
                    x2_disp = x2 / scale_x
                    y2_disp = y2 / scale_y

                    canvas.create_rectangle(x1_disp, y1_disp, x2_disp, y2_disp, outline='green', tags=('selection'))

    def _setup_selection_canvas(self, canvas, img_tk, scale_x=1, scale_y=1, monitor=None):
        canvas.pack(fill=tk.BOTH, expand=True)
        canvas.image = img_tk
        canvas.create_image(0, 0, image=img_tk, anchor=tk.NW)

        canvas_info = {
            'canvas': canvas,
            'scale_x': scale_x,
            'scale_y': scale_y,
            'monitor': monitor
        }
        self.canvases.append(canvas_info)

        start_x, start_y, rect = None, None, None

        def on_click(event):
            self.drawing = True
            nonlocal start_x, start_y, rect
            start_x, start_y = event.x, event.y
            rect = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline='red', tags=('selection'))

        def on_drag(event):
            nonlocal start_x, start_y, rect
            if rect:
                canvas.coords(rect, start_x, start_y, event.x, event.y)

        def on_release(event):
            nonlocal start_x, start_y, rect
            if start_x is None or start_y is None:
                return

            end_x, end_y = event.x, event.y
            x1 = int(min(start_x, end_x) * scale_x)
            y1 = int(min(start_y, end_y) * scale_y)
            x2 = int(max(start_x, end_x) * scale_x)
            y2 = int(max(start_y, end_y) * scale_y)

            rect = None
            start_x = None
            start_y = None
            self.drawing = False
            self.add_selection(monitor, (x1, y1, x2, y2))

        def reset_selection(event):
            nonlocal start_x, start_y, rect
            if rect:
                canvas.delete(rect)
                rect = None

            start_x = None
            start_y = None
            self.drawing = False

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
            if (monitor['width'] >= original_width and monitor['height'] >= original_height):
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
                self.result_queue.put(False)
                continue

            self.root = tk.Tk()

            if not self.mac_init_done and sys.platform == 'darwin':
                app = NSApplication.sharedApplication()
                app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
                self.mac_init_done = True

            self.root.withdraw()
            self.after_id = self.root.after(50, self.process_keyboard_events)

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

    result = None
    while result is None and selector_process.is_alive():
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
