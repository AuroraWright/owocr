import sys
import multiprocessing
import queue
import threading
import ctypes
import importlib.resources

from PIL import Image
import pystrayfix

if sys.platform == 'darwin':
    from AppKit import NSApplication, NSApplicationActivationPolicyAccessory


class TrayGUI:
    def __init__(self, engine_names, selected_engine, paused, screen_capture_enabled, result_queue, command_queue):
        self.engine_names = engine_names
        self.paused = paused
        self.current_engine_index = selected_engine
        self.screen_capture_enabled = screen_capture_enabled
        self.command_queue = command_queue
        self.result_queue = result_queue
        self.terminated = False
        self.icon = None
        self.comm_thread = None

    def load_icon_image(self):
        if sys.platform == 'darwin':
            icon_name = 'mac_tray_icon.png'
        else:
            icon_name = 'icon.png'
        icon_path = importlib.resources.files(__name__).joinpath('data', icon_name)
        return Image.open(icon_path)

    def setup_menu(self):
        pause_item = pystrayfix.MenuItem(
            lambda item: 'Unpause' if self.paused else 'Pause',
            self.on_pause_clicked
        )

        def make_action_func(k):
            def func(icon, item):
                self.on_engine_clicked(k)
            return func

        def make_checked_func(k):
            def func(item):
                return self.current_engine_index == k
            return func

        engine_menu_items = []
        for i, name in enumerate(self.engine_names):
            engine_menu_items.append(
                pystrayfix.MenuItem(name, make_action_func(i), checked=make_checked_func(i))
            )

        engine_menu = pystrayfix.Menu(*engine_menu_items)
        capture_item = pystrayfix.MenuItem(
            'Take a screenshot',
            self.on_capture_clicked,
            visible=self.screen_capture_enabled
        )
        capture_area_selection_item = pystrayfix.MenuItem(
            'Capture area selection',
            self.on_capture_area_selector_clicked,
            visible=self.screen_capture_enabled
        )

        menu = pystrayfix.Menu(
            pause_item,
            pystrayfix.MenuItem('Change engine', engine_menu),
            capture_item,
            capture_area_selection_item,
            pystrayfix.Menu.SEPARATOR,
            pystrayfix.MenuItem('Quit', self.on_quit_clicked)
        )

        return menu

    def send_to_main(self, action, data = None):
        self.result_queue.put((action, data))

    def receive_from_main(self):
        while not self.terminated:
            try:
                data = self.command_queue.get(timeout=0.2)
                self.handle_main_message(data)
            except queue.Empty:
                continue

    def handle_main_message(self, message):
        action, data = message
        if action == 'update_pause':
            self.paused = data
            self.icon.update_menu()
        elif action == 'update_engine':
            self.current_engine_index = data
            self.icon.update_menu()
        elif action == 'terminate':
            self.terminated = True
            self.icon.stop()

    def on_pause_clicked(self, icon, item):
        self.paused = not self.paused
        self.send_to_main('toggle_pause')
        icon.update_menu()

    def on_capture_clicked(self, icon, item):
        self.send_to_main('capture')

    def on_capture_area_selector_clicked(self, icon, item):
        self.send_to_main('capture_area_selector')

    def on_engine_clicked(self, engine_index):
        if engine_index != self.current_engine_index:
            self.current_engine_index = engine_index
            self.send_to_main('change_engine', engine_index)
            self.icon.update_menu()

    def on_quit_clicked(self, icon, item):
        self.send_to_main('terminate')
        self.terminated = True
        icon.stop()

    def run(self):
        if sys.platform == 'win32':
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        elif sys.platform == 'darwin':
            app = NSApplication.sharedApplication()
            app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

        self.comm_thread = threading.Thread(target=self.receive_from_main, daemon=True)
        self.comm_thread.start()
        self.icon = pystrayfix.Icon('owocr', self.load_icon_image(), 'owocr', self.setup_menu())
        self.send_to_main('started')
        self.icon.run()
        self.comm_thread.join()


def run_tray_gui(engine_names, selected_engine, paused, screen_capture_enabled, result_queue, command_queue):
    tray = TrayGUI(engine_names, selected_engine, paused, screen_capture_enabled, result_queue, command_queue)
    tray.run()

tray_process = None

def start_tray_process(engine_names, selected_engine, paused, screen_capture_enabled, result_queue, command_queue):
    global tray_process

    tray_process = multiprocessing.Process(target=run_tray_gui, args=(engine_names, selected_engine, paused, screen_capture_enabled, result_queue, command_queue))
    tray_process.daemon = True
    tray_process.start()

    result = None
    while result is None and tray_process.is_alive():
        try:
            result = result_queue.get(timeout=0.1)
        except:
            continue

def terminate_tray_process_if_running(command_queue):
    if tray_process and tray_process.is_alive():
        command_queue.put(('terminate', None))
        tray_process.join()
