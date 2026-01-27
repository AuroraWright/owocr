import multiprocessing
import queue
import ctypes
import mss
from loguru import logger
from PIL import Image
from pynputfix import keyboard
import sys
import os

if sys.platform == 'darwin':
    from AppKit import NSApplication, NSApplicationActivationPolicyAccessory

try:
    from PIL import ImageTk
    import tkinter as tk
    selector_available = True
except:
    selector_available = False


class ScreenSelector:
    def __init__(self, result_queue, command_queue):
        self.root = None
        self.result_queue = result_queue
        self.command_queue = command_queue
        self.ctrl_pressed = False
        self.drawing = False
        self.windows = []
        self.canvases = []
        self.selections = []
        self.previous_coordinates = []
        self.keyboard_event_queue = queue.Queue()
        self.real_monitors = []

    def start_key_listener(self):
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release
        )
        self.keyboard_listener.start()
        if sys.platform != 'linux':
            self.keyboard_listener.wait()

    def stop_key_listener(self):
        self.keyboard_listener.stop()
        if sys.platform != 'linux':
            self.keyboard_listener.join()

    def on_key_press(self, key):
        if not self.windows:
            return

        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r, keyboard.Key.cmd_l, keyboard.Key.cmd_r):
            self.ctrl_pressed = True
            self.keyboard_event_queue.put('show_previous_selections')

    def on_key_release(self, key):
        if not self.windows:
            return

        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r, keyboard.Key.cmd_l, keyboard.Key.cmd_r):
            self.ctrl_pressed = False
            self.keyboard_event_queue.put('return_selections')

        elif key == keyboard.Key.backspace:
            self.keyboard_event_queue.put('clear_selections')

        elif key == keyboard.Key.esc:
            self.keyboard_event_queue.put('return_empty')

    def process_events(self):
        if self.windows:
            try:
                while True:
                    event_type = self.keyboard_event_queue.get_nowait()
                    if event_type == 'return_selections':
                        if self.selections and not self.drawing:
                            self.return_all_selections()
                    elif event_type == 'return_empty':
                        self.return_empty()
                    elif event_type == 'clear_selections':
                        self.clear_all_selections()
                    elif event_type == 'show_previous_selections':
                        self.show_previous_selections()
            except queue.Empty:
                pass
        else:
            command = self.command_queue.get()
            image, coordinates, is_window = command

            if image == False:
                self.root.destroy()
                return
            elif image == True:
                self.result_queue.put(False)
            else:
                self.previous_coordinates = coordinates if coordinates else []

                monitors, window_scale, images = image
                if is_window:
                    self.find_monitor_and_create_window(monitors, window_scale, images, None)
                else:
                    for i, monitor in enumerate(monitors):
                        if self.real_monitors:
                            self.find_monitor_and_create_window(self.real_monitors, 1, images[i], monitor)
                        else:
                            self.create_window(monitor, images[i])

        self.root.after(50, self.process_events)

    def add_selection(self, monitor, scale_x, scale_y, coordinates):
        ctrl_pressed = self.ctrl_pressed
        x1, y1, x2, y2 = coordinates

        if x1 == x2 or y1 == y2:
            if ctrl_pressed:
                return
            coordinates = None

        valid_selection = True
        if coordinates:
            to_remove = []
            for i, selection in enumerate(self.selections):
                if selection['monitor'] == monitor:
                    x1_old, y1_old, x2_old, y2_old = selection['coordinates']
                    if x1_old <= x1 and y1_old <= y1 and x2_old >= x2 and y2_old >= y2:
                        valid_selection = False
                        break
                    if x1 <= x1_old and y1 <= y1_old and x2 >= x2_old and y2 >= y2_old:
                        to_remove.append(i)
            if valid_selection:
                for i in sorted(to_remove, reverse=True):
                    self.selections.pop(i)

        if valid_selection:
            self.selections.append({
                'monitor': monitor,
                'scale_x': scale_x,
                'scale_y': scale_y,
                'coordinates': coordinates
            })

        if not ctrl_pressed:
            self.keyboard_event_queue.put('return_selections')
            return

        self.redraw_selections()

    def remove_selection(self, monitor, coordinates):
        x1, y1, x2, y2 = coordinates
        for i, selection in enumerate(self.selections):
            if selection['monitor'] == monitor and selection['coordinates'] == (x1, y1, x2, y2):
                self.selections.pop(i)
                self.redraw_selections()
                break

    def clear_all_selections(self):
        if self.drawing:
            return

        self.selections.clear()
        self.previous_coordinates.clear()
        self.redraw_selections()

    def return_empty(self):
        self.result_queue.put(False)
        self.destroy_all_windows()

    def destroy_all_windows(self):
        for window in self.windows:
            window.destroy()

        self.selections.clear()
        self.canvases.clear()
        self.windows.clear()
        self.ctrl_pressed = False
        self.drawing = False
        while not self.keyboard_event_queue.empty():
            self.keyboard_event_queue.get()

    def cleanup_ui(self):
        self.root.update()
        self.root = None

    def return_all_selections(self):
        selections_abs = []
        for selection in self.selections:
            monitor = selection['monitor']
            coordinates = selection['coordinates']
            scale_x = selection['scale_x']
            scale_y = selection['scale_y']

            if coordinates:
                scaled_x1 = int(scale_x * coordinates[0])
                scaled_y1 = int(scale_y * coordinates[1])
                scaled_x2 = int(scale_x * coordinates[2])
                scaled_y2 = int(scale_y * coordinates[3])
                coordinates = (scaled_x1, scaled_y1, scaled_x2, scaled_y2)

            if monitor and coordinates:
                abs_x1 = monitor['left'] + coordinates[0]
                abs_y1 = monitor['top'] + coordinates[1]
                abs_x2 = monitor['left'] + coordinates[2]
                abs_y2 = monitor['top'] + coordinates[3]
                coordinates = (abs_x1, abs_y1, abs_x2, abs_y2)

            selections_abs.append({
                'monitor': monitor,
                'coordinates': coordinates
            })

        self.result_queue.put(selections_abs)
        self.destroy_all_windows()

    def show_previous_selections(self):
        if not self.previous_coordinates:
            return
        if self.selections:
            self.previous_coordinates = []
            return
        if self.drawing:
            self.previous_coordinates = []
            return

        for prev_sel in self.previous_coordinates:
            monitor = prev_sel['monitor']
            abs_coords = prev_sel['coordinates']

            canvas_info = None
            if not monitor:
                canvas_info = self.canvases[0]
            else:
                for info in self.canvases:
                    if info['monitor'] == monitor:
                        canvas_info = info
                        break

            if not canvas_info:
                continue

            scale_x = canvas_info['scale_x']
            scale_y = canvas_info['scale_y']

            if monitor:
                rel_x1 = abs_coords[0] - monitor['left']
                rel_y1 = abs_coords[1] - monitor['top']
                rel_x2 = abs_coords[2] - monitor['left']
                rel_y2 = abs_coords[3] - monitor['top']
            else:
                rel_x1, rel_y1, rel_x2, rel_y2 = abs_coords

            canvas_x1 = rel_x1 / scale_x
            canvas_y1 = rel_y1 / scale_y
            canvas_x2 = rel_x2 / scale_x
            canvas_y2 = rel_y2 / scale_y

            self.selections.append({
                'monitor': monitor,
                'scale_x': scale_x,
                'scale_y': scale_y,
                'coordinates': (canvas_x1, canvas_y1, canvas_x2, canvas_y2)
            })

        self.previous_coordinates = []
        self.redraw_selections()

    def redraw_selections(self):
        for canvas_info in self.canvases:
            canvas = canvas_info['canvas']
            monitor = canvas_info['monitor']

            items = canvas.find_all()
            for item in items:
                if canvas.gettags(item) and 'selection' in canvas.gettags(item):
                    canvas.delete(item)

            for selection in self.selections:
                if selection['monitor'] == monitor:
                    x1, y1, x2, y2 = selection['coordinates']
                    canvas.create_rectangle(x1, y1, x2, y2, outline='green', tags=('selection', 'permanent'))

    def _setup_selection_canvas(self, canvas, img_tk, scale_x, scale_y, monitor):
        canvas.pack(fill=tk.BOTH, expand=True)
        canvas.image = img_tk
        canvas.create_image(0, 0, image=img_tk, anchor=tk.NW)

        canvas_info = {
            'canvas': canvas,
            'monitor': monitor,
            'scale_x': scale_x,
            'scale_y': scale_y
        }
        self.canvases.append(canvas_info)

        start_x, start_y, rect = None, None, None

        def on_click(event):
            self.drawing = True
            nonlocal start_x, start_y, rect
            start_x, start_y = event.x, event.y
            rect = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline='red', tags=('selection', 'temporary'))

        def on_drag(event):
            nonlocal start_x, start_y, rect
            if rect:
                canvas.coords(rect, start_x, start_y, event.x, event.y)

        def on_release(event):
            nonlocal start_x, start_y, rect
            if start_x is None or start_y is None:
                return

            end_x, end_y = event.x, event.y
            x1 = min(start_x, end_x)
            y1 = min(start_y, end_y)
            x2 = max(start_x, end_x)
            y2 = max(start_y, end_y)

            rect = None
            start_x = None
            start_y = None
            self.drawing = False

            if x1 == x2 or y1 == y2:
                items = canvas.find_all()
                for item in items:
                    tags = canvas.gettags(item)
                    if tags and 'permanent' in tags:
                        coords = canvas.coords(item)
                        x1_item, y1_item, x2_item, y2_item = coords
                        if x1_item <= end_x <= x2_item and y1_item <= end_y <= y2_item:
                            self.remove_selection(monitor, coords)
                            return

                self.clear_all_selections()

            self.add_selection(monitor, scale_x, scale_y, (x1, y1, x2, y2))

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

    def _create_selection_window(self, img, geometry, scale_x, scale_y, monitor):
        window = tk.Toplevel(self.root)
        window.geometry(geometry)
        window.overrideredirect(1)
        window.attributes('-topmost', 1)
        self.windows.append(window)

        img_tk = ImageTk.PhotoImage(img)
        canvas = tk.Canvas(window, cursor='cross', highlightthickness=0)

        self._setup_selection_canvas(canvas, img_tk, scale_x, scale_y, monitor)

    def find_monitor_and_create_window(self, monitors, window_scale, img, fake_monitor):
        if window_scale != 1:
            img = img.resize((int(img.width * window_scale), int(img.height * window_scale)), Image.Resampling.LANCZOS)
            scale_x = 1 / window_scale
            scale_y = 1 / window_scale
        else:
            scale_x = 1
            scale_y = 1

        original_width, original_height = img.size
        display_monitor = None

        for monitor in monitors:
            if (monitor['width'] == original_width and monitor['height'] == original_height):
                display_monitor = monitor
                break

        if not display_monitor:
            for monitor in monitors:
                if (monitor['width'] > original_width and monitor['height'] > original_height):
                    display_monitor = monitor
                    break

        if not display_monitor:
            display_monitor = monitors[0]

        window_width = min(original_width, display_monitor['width'])
        window_height = min(original_height, display_monitor['height'])
        left = display_monitor['left'] + (display_monitor['width'] - window_width) // 2
        top = display_monitor['top'] + (display_monitor['height'] - window_height) // 2

        geometry = f"{window_width}x{window_height}+{left}+{top}"

        if img.width > window_width or img.height > window_height:
            img = img.resize((window_width, window_height), Image.Resampling.LANCZOS)
            scale_x *= original_width / window_width
            scale_y *= original_height / window_height

        self._create_selection_window(img, geometry, scale_x, scale_y, fake_monitor)

    def create_window(self, monitor, img):
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
        self.start_key_listener()

        if sys.platform == 'linux' and os.environ.get('XDG_SESSION_TYPE', '').lower() == 'wayland':
            self.real_monitors = mss.mss().monitors[1:]
        elif sys.platform == 'win32':
            ctypes.windll.shcore.SetProcessDpiAwareness(2)

        self.root = tk.Tk()
        self.root.withdraw()

        if sys.platform == 'darwin':
            app = NSApplication.sharedApplication()
            app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

        self.root.after(0, self.process_events)
        self.root.mainloop()
        self.cleanup_ui()
        self.stop_key_listener()


def run_screen_selector(result_queue, command_queue):
    selector = ScreenSelector(result_queue, command_queue)
    selector.start()

selector_process = None
result_queue = None
command_queue = None

def get_screen_selection(image_data, previous_coordinates, is_window, permanent_process):
    global selector_process, result_queue, command_queue

    if not selector_available:
        logger.error('tkinter or PIL with tkinter support are not installed, unable to open picker')
        sys.exit(1)

    if selector_process is None or not selector_process.is_alive():
        result_queue = multiprocessing.Queue()
        command_queue = multiprocessing.Queue()
        selector_process = multiprocessing.Process(target=run_screen_selector, args=(result_queue, command_queue), daemon=True)
        selector_process.start()

    command_queue.put((image_data, previous_coordinates, is_window))

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
        command_queue.put((False, None, None))
        selector_process.join()
