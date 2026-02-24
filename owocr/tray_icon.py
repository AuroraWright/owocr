import sys
import multiprocessing
import queue
import threading
import inspect
import importlib.resources


class GlobalImport:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        caller_frame = inspect.getouterframes(inspect.currentframe())[1].frame
        collector = inspect.getargvalues(caller_frame).locals
        caller_frame.f_globals.update(collector)

class TrayGUI:
    def __init__(self, result_queue, command_queue):
        with GlobalImport():
            import pystrayfix
            from PIL import Image, ImageDraw
        self.enabled = False
        self.error = False
        self.command_queue = command_queue
        self.result_queue = result_queue
        self.terminated = False
        self.icon = None
        self.comm_thread = None
        self.normal_icon = self.load_icon_image()
        self.paused_icon = self.create_paused_icon()
        self.stopped_icon = self.create_stopped_icon()
        self.is_bundled = getattr(sys, 'frozen', False)

    def create_paused_icon(self):
        r, g, b, a = self.normal_icon.split()

        def process_alpha(value):
            if value == 255:
                return 85
            return value

        a_processed = a.point(process_alpha)
        result = Image.merge('RGBA', (r, g, b, a_processed))
        return result

    def create_stopped_icon(self):
        r, g, b, a = self.paused_icon.split()
        width, height = self.paused_icon.size

        alpha_draw = ImageDraw.Draw(a)
        line_width = width // 12

        # Top-left to bottom-right line
        alpha_draw.line([(0, 0), (width, height)], fill=0, width=line_width)

        # Top-right to bottom-left line
        alpha_draw.line([(width, 0), (0, height)], fill=0, width=line_width)

        return Image.merge('RGBA', (r, g, b, a))

    def load_icon_image(self):
        if sys.platform == 'darwin':
            icon_name = 'mac_tray_icon.png'
        else:
            icon_name = 'icon.png'
        icon_path = importlib.resources.files(__name__).joinpath('data', icon_name)
        return Image.open(icon_path)

    def setup_menu(self):
        launch_config_item = pystrayfix.MenuItem('Configure', self.on_config_launch_clicked)
        launch_log_viewer_item = pystrayfix.MenuItem('View log', self.on_log_viewer_launch_clicked, visible=self.is_bundled)
        quit_item = pystrayfix.MenuItem('Quit', self.on_quit_clicked)

        if not self.enabled:     
            status_item = pystrayfix.MenuItem(lambda item: 'Starting...' if not self.error else 'An error occurred!', self.on_log_viewer_launch_clicked)

            menu = pystrayfix.Menu(
                status_item,
                launch_config_item,
                launch_log_viewer_item,
                pystrayfix.Menu.SEPARATOR,
                quit_item
            )
        else:
            pause_item = pystrayfix.MenuItem(lambda item: 'Unpause' if self.paused else 'Pause', self.on_pause_clicked, default=pystrayfix.Icon.HAS_DEFAULT_ACTION)

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
                engine_menu_items.append(pystrayfix.MenuItem(name, make_action_func(i), checked=make_checked_func(i)))

            engine_menu = pystrayfix.Menu(*engine_menu_items)
            capture_item = pystrayfix.MenuItem('Take a screenshot', self.on_capture_clicked, visible=self.screen_capture_enabled or self.obs_enabled)
            capture_area_selection_item = pystrayfix.MenuItem('Select capture area', self.on_capture_area_selector_clicked, visible=self.screen_capture_enabled)

            menu = pystrayfix.Menu(
                pause_item,
                pystrayfix.MenuItem('Change engine', engine_menu),
                capture_item,
                capture_area_selection_item,
                launch_config_item,
                launch_log_viewer_item,
                pystrayfix.Menu.SEPARATOR,
                quit_item
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
        if action == 'enable':
            self.enabled = True
            self.engine_names, self.current_engine_index, self.paused, self.screen_capture_enabled, self.obs_enabled = data
            self.icon.icon = self.paused_icon if self.paused else self.normal_icon
            self.icon.menu = self.setup_menu()
            self.icon.update_menu()
        elif action == 'error':
            self.error = True
            self.enabled = False
            self.icon.menu = self.setup_menu()
            self.icon.icon = self.stopped_icon
            self.icon.update_menu()
        elif action == 'update_pause':
            self.paused = data
            self.icon.icon = self.paused_icon if self.paused else self.normal_icon
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
        icon.icon = self.paused_icon if self.paused else self.normal_icon
        icon.update_menu()

    def on_capture_clicked(self, icon, item):
        self.send_to_main('capture')

    def on_capture_area_selector_clicked(self, icon, item):
        self.send_to_main('capture_area_selector')

    def on_config_launch_clicked(self, icon, item):
        self.send_to_main('launch_config')

    def on_log_viewer_launch_clicked(self, icon, item):
        self.send_to_main('launch_log_viewer')

    def on_engine_clicked(self, engine_index):
        if engine_index != self.current_engine_index:
            self.current_engine_index = engine_index
            self.send_to_main('change_engine', engine_index)
            self.icon.update_menu()

    def on_quit_clicked(self, icon, item):
        self.send_to_main('terminate')
        self.terminated = True
        icon.update_menu = lambda: None
        icon.stop()

    def run(self):
        if sys.platform == 'win32':
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        elif sys.platform == 'darwin':
            from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
            app = NSApplication.sharedApplication()
            app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

        self.comm_thread = threading.Thread(target=self.receive_from_main, daemon=True)
        self.comm_thread.start()
        self.icon = pystrayfix.Icon('owocr', self.stopped_icon, 'owocr', self.setup_menu())
        self.send_to_main('started')
        self.icon.run()
        self.comm_thread.join()


def run_tray_gui(result_queue, command_queue):
    tray = TrayGUI(result_queue, command_queue)
    tray.run()

tray_process = None

def start_minimal_tray(result_queue, command_queue):
    global tray_process

    tray_process = multiprocessing.Process(target=run_tray_gui, args=(result_queue, command_queue), daemon=True)
    tray_process.start()

    result = None
    while result is None and tray_process.is_alive():
        try:
            result = result_queue.get(timeout=0.1)
        except:
            continue

def start_full_tray(result_queue, command_queue, engine_names, selected_engine, paused, screen_capture_enabled, obs_enabled):
    if not tray_process:
        start_minimal_tray(result_queue, command_queue)
    command_queue.put(('enable', (engine_names, selected_engine, paused, screen_capture_enabled, obs_enabled)))

def wait_for_tray_process():
    if tray_process and tray_process.is_alive():
        tray_process.join()

def terminate_tray_process_if_running(command_queue):
    if tray_process and tray_process.is_alive():
        command_queue.put(('terminate', None))
        tray_process.join()
