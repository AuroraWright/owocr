import configparser
import os
import inspect
import sys
import time
import importlib.resources

from .ocr import *
from .config import Config

if sys.platform == 'win32':
    import ctypes

try:
    import tkinter as tk
    from tkinter import PhotoImage, ttk, messagebox, filedialog
    editor_available = True
except:
    editor_available = False


class GlobalImport:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.collector = inspect.getargvalues(inspect.getouterframes(inspect.currentframe())[1].frame).locals
        globals().update(self.collector)

class HotkeyRecorder:
    def __init__(self):
        with GlobalImport():
            from pynputfix import keyboard
        self.listener = None
        self.recording = False
        self.current_keys = set()
        self.current_pynput_keys = set()
        self.window_scale = 1
        self.popup = None
        self.on_hotkey_recorded = None
        self.cancelled = False

    def start_listener(self):
        def on_press(key):
            if not self._should_handle_event():
                return

            key_str = None
            try:
                key_str = self.listener.canonical(key).char
            except AttributeError:
                pass
            if not key_str:
                key_str = f'<{str(key)[4:]}>'

            self.current_pynput_keys.add(key_str)
            self.current_keys.add(key_str)
            self._update_popup_label()

        def on_release(key):
            if not self._should_handle_event():
                return

            key_str = None
            try:
                key_str = self.listener.canonical(key).char
            except AttributeError:
                pass
            if not key_str:
                key_str = f'<{str(key)[4:]}>'

            self.current_pynput_keys.discard(key_str)

            if not self.current_pynput_keys:
                self.finish_recording()

        self.listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release
        )
        self.listener.start()
        if sys.platform != 'linux':
            self.listener.wait()

    def stop_listener(self):
        self.listener.stop()
        if sys.platform != 'linux':
            self.listener.join()

    def _should_handle_event(self):
        if not self.recording:
            return False
        if not self.popup or not self.popup.winfo_exists():
            self.stop_recording()
            return False
        return True

    def start_recording(self, parent_widget):
        if self.recording:
            if self.popup:
                self.popup.focus_force()
            return

        self.recording = True
        self.current_keys = set()
        self.current_pynput_keys = set()
        self.cancelled = False

        self._create_popup_window(parent_widget)
        self._update_popup_label()
        self.popup.focus_force()

    def _create_popup_window(self, parent_widget):
        self.popup = tk.Toplevel(parent_widget)
        self.popup.title('Record Hotkey')

        width = int(300 * self.window_scale)
        height = int(100 * self.window_scale)
        self.popup.geometry(f'{width}x{height}')
        self.popup.resizable(False, False)

        self._center_popup(parent_widget, width, height)
        self._make_popup_modal(parent_widget)
        self._create_popup_content()

    def _center_popup(self, parent_widget, width, height):
        popup_x = parent_widget.winfo_rootx() + parent_widget.winfo_width() // 2 - width // 2
        popup_y = parent_widget.winfo_rooty() + parent_widget.winfo_height() // 2 - height // 2
        self.popup.geometry(f'+{popup_x}+{popup_y}')

    def _make_popup_modal(self, parent_widget):
        self.popup.transient(parent_widget)
        self.popup.grab_set()

    def _create_popup_content(self):
        self.label_var = tk.StringVar(value='Press any key combination...\n(Click to cancel)')
        self.label = ttk.Label(self.popup, textvariable=self.label_var, justify=tk.CENTER, anchor=tk.CENTER)
        self.label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        for widget in [self.popup, self.label]:
            widget.bind('<Button-1>', lambda _: self.cancel_recording())

    def _update_popup_label(self):
        if self.current_keys:
            keys = sorted(self.current_keys)
            hotkey_str = '+'.join(keys)
            text = f'Current: {hotkey_str}\n(Release all keys to finish)'
        else:
            text = 'Press any key combination...\n(Click to cancel)'

        self.label_var.set(text)

    def finish_recording(self):
        if self.current_keys and not self.cancelled and self.on_hotkey_recorded:
            keys = sorted(self.current_keys)
            hotkey_str = '+'.join(keys)
            self.on_hotkey_recorded(hotkey_str)

        self.stop_recording()

    def cancel_recording(self):
        self.cancelled = True
        self.stop_recording()

    def stop_recording(self):
        self.recording = False
        self.current_keys.clear()
        self.current_pynput_keys.clear()
        if self.popup and self.popup.winfo_exists():
            self.popup.grab_release()
            self.popup.destroy()
            self.popup = None


class ConfigGUI:
    def __init__(self, root):
        self.root = root
        self.config_path = Config.config_path
        self.is_bundled = getattr(sys, 'frozen', False)

        self._setup_window()
        self._initialize_styles()

        self.categories = {}
        self.widgets = {}
        self.default_values = {}
        self.engine_config_widgets = {}
        self.tab_canvases = {}
        self.tab_scrollable_frames = {}
        self.tab_scrollbars = {}

        self._load_categories_and_defaults()
        self._get_engine_info()
        self._create_widgets()
        self._load_config()

    def _setup_window(self):
        self.root.title('owocr Configuration Editor')
        icon_path = importlib.resources.files(__name__).joinpath('data', 'icon.png')
        icon = PhotoImage(file=icon_path)
        self.root.iconphoto(True, icon)

        width, height = 700, 700
        window_scale = 1.0
        if sys.platform == 'win32':
            hwnd = self.root.winfo_id()
            dpi = ctypes.windll.user32.GetDpiForWindow(hwnd)
            window_scale = dpi / 96.0
            hotkey_recorder.window_scale = window_scale

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
        font_family = 'Segoe UI' if sys.platform == 'win32' else 'TkTextFont'
        style.configure('Category.TLabelframe', font=(font_family, 11, 'bold'))
        style.configure('Help.TLabel', font=(font_family, 9), foreground='gray')
        style.configure('TNotebook.Tab', padding=(10, 3))

    def _load_categories_and_defaults(self):
        self.default_values = Config.default_config
        self.general_config_options = {
            'General': [
                ('read_from', 'dropdown', 'Input source', ['clipboard', 'websocket', 'unixsocket', 'screencapture']),
                ('read_from_secondary', 'dropdown', 'Optional secondary input source', ['', 'clipboard', 'websocket', 'unixsocket', 'screencapture']),
                ('write_to', 'dropdown', 'Output destination', ['clipboard', 'websocket']),
                ('websocket_port', 'int', 'Websocket port'),
                ('delay_seconds', 'float', 'Check the clipboard/directory every X seconds (ignored on Windows/Wayland)'),
                ('delete_images', 'bool', 'Delete images from the folder after processing'),
                ('pause_at_startup', 'bool', 'Pause when owocr starts'),
                ('notifications', 'bool', 'Show OS notifications with the detected text'),
                ('tray_icon', 'bool', 'Show an OS tray icon to change the engine, pause/unpause,\nchange the screen capture area selection, take a screenshot\nand launch this configuration'),
                ('auto_pause', 'float', 'Automatically pause after X seconds of inactivity (0 to disable)'),
                ('output_format', 'dropdown', 'Output format', ['text', 'json']),
                ('verbosity', 'int', 'Terminal verbosity level:\n-2: show everything\n-1: only timestamps\n0: only errors\nGreater than 0: maximum amount of characters'),
            ],
            'Engines': [
                ('engines', 'multicheckbox', 'List of enabled OCR engines'),
                ('engine', 'dropdown', 'Primary OCR engine'),
                ('engine_secondary', 'dropdown', 'Optional secondary OCR engine for screen capture'),
            ],
            'Screen capture': [
                ('screen_capture_area', 'special_screen_capture_area', 'Area to capture'),
                ('screen_capture_window_area', 'special_screen_capture_window_area', 'Window subsection to capture'),
                ('screen_capture_only_active_windows', 'bool', "Only capture a window while it's not in the background"),
                ('screen_capture_delay_seconds', 'float', 'Capture every X seconds (-1 to disable continuous capture)'),
                ('screen_capture_frame_stabilization', 'float', 'Wait X seconds until text is stable:\n-1: wait for two frames\n0: disable (faster, only works when text is shown all at once)'),
                ('screen_capture_line_recovery', 'bool', 'Try to recover lines missed by frame stabilization (can increase glitches)'),
                ('screen_capture_regex_filter', 'str', 'Regex filter for unwanted text'),
                ('language', 'dropdown', 'Language code, used to cleanup text', ['ja', 'en', 'zh', 'ko', 'ar', 'ru', 'el', 'he', 'th']),
            ],
            'Hotkeys': [
                ('combo_pause', 'str', 'Pause/resume hotkey'),
                ('combo_engine_switch', 'str', 'OCR engine switch hotkey'),
                ('screen_capture_combo', 'str', 'Capture hotkey'),
                ('coordinate_selector_combo', 'str', 'Screen capture area selection hotkey'),
            ],
            'Processing': [
                ('join_lines', 'bool', 'Join lines without separators'),
                ('join_paragraphs', 'bool', 'Join paragraphs without separators'),
                ('line_separator', 'str', 'Custom line separator (supports special characters like \\n for a newline)'),
                ('paragraph_separator', 'str', 'Custom paragraph separator (supports special characters like \\n for a newline)'),
                ('reorder_text', 'bool', 'Regroup and reorder text. If disabled, text is shown as-is from the OCR engine'),
                ('furigana_filter', 'bool', 'Filter out furigana lines for Japanese if reorder_text is enabled'),
            ],
            'Advanced': [
                ('screen_capture_old_macos_api', 'bool', 'Use old macOS screen capture API'),
                ('wayland_use_wlclipboard', 'bool', 'Use wl-clipboard on Linux/Wayland'),
            ]
        }
        self.engine_config_options = {
            'winrtocr': [
                ('url', 'str', 'URL for WinRT OCR service (ignored on Windows)', 'http://aaa.xxx.yyy.zzz:8000'),
            ],
            'oneocr': [
                ('url', 'str', 'URL for OneOCR service (ignored on Windows)', 'http://aaa.xxx.yyy.zzz:8001'),
            ],
            'azure': [
                ('api_key', 'str', 'Azure API key', 'api_key_here'),
                ('endpoint', 'str', 'Azure endpoint', 'https://YOURPROJECT.cognitiveservices.azure.com/'),
            ],
            'mangaocr': [
                ('pretrained_model_name_or_path', 'str', 'Model name or path', 'kha-white/manga-ocr-base'),
                ('force_cpu', 'bool', 'Force CPU usage', False),
            ],
            'easyocr': [
                ('gpu', 'bool', 'Use GPU if available', True),
            ],
            'ocrspace': [
                ('api_key', 'str', 'OCR.space API key', 'api_key_here'),
                ('engine_version', 'int', 'Engine version (1 or 2)', 2),
            ],
            'rapidocr': [
                ('high_accuracy_detection', 'bool', 'Use high accuracy detection', False),
                ('high_accuracy_recognition', 'bool', 'Use high accuracy recognition', True),
            ],
            'avision': [
                ('fast_mode', 'bool', 'Use fast mode', False),
                ('language_correction', 'bool', 'Enable language correction', True),
            ]
        }

    def _get_engine_info(self):
        self.all_engines = []
        self.secondary_engines = []
        self.engine_name_to_class = {}
        self.engine_class_to_display = {}
        self.config_entry_to_engines = {}
        for _, engine_class in inspect.getmembers(sys.modules[__name__], self._is_engine_class):
            self._process_engine_class(engine_class)

    def _is_engine_class(self, obj):
        return inspect.isclass(obj) and hasattr(obj, '__module__') and obj.__module__ and __package__ + '.ocr' in obj.__module__ and hasattr(obj, 'name')

    def _process_engine_class(self, engine_class):
        internal_name = engine_class.name
        display_name = engine_class.readable_name

        self.all_engines.append(display_name)
        self.engine_name_to_class[display_name] = engine_class
        self.engine_class_to_display[internal_name] = display_name

        config_entry = engine_class.config_entry

        if config_entry is not None:
            if config_entry not in self.config_entry_to_engines:
                self.config_entry_to_engines[config_entry] = []
            self.config_entry_to_engines[config_entry].append(display_name)

        if engine_class.local and engine_class.coordinate_support:
            self.secondary_engines.append(display_name)

    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        if sys.platform == 'darwin':
            notebook_padding = 0
            status_bar_padding = 20
        else:
            notebook_padding = 10
            status_bar_padding = 10

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, notebook_padding))

        self._create_category_tabs()
        self._create_buttons(main_frame)

        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(main_frame, textvariable=self.status_var,relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(status_bar_padding, 0))

        self._configure_grid_weights(main_frame)

    def _create_category_tabs(self):
        self.tabs = {}
        for category, options in self.general_config_options.items():
            tab = ttk.Frame(self.notebook)
            self.notebook.add(tab, text=category)
            self.tabs[category] = tab
            self._create_scrollable_tab(tab, category, options)

    def _create_scrollable_tab(self, parent, category, options):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        self.tab_canvases[category] = canvas
        self.tab_scrollable_frames[category] = scrollable_frame
        self.tab_scrollbars[category] = scrollbar

        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor=tk.NW, tags='scrollable_frame')
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))

        self._configure_canvas_scrolling(canvas, scrollbar, scrollable_frame, canvas_window)

        row = 0
        for option_data in options:
            row = self._create_option_widget(scrollable_frame, option_data, row, category)

        if category == 'Engines':
            row = self._add_engine_configuration_sections(scrollable_frame, row)

        scrollable_frame.columnconfigure(1, weight=1)

        return canvas

    def _configure_canvas_scrolling(self, canvas, scrollbar, scrollable_frame, canvas_window):
        def update_scrollable_frame_height(event):
            canvas_height = event.height
            canvas_width = event.width

            scrollable_frame.update_idletasks()
            frame_height = scrollable_frame.winfo_reqheight()

            if frame_height <= canvas_height:
                scrollbar.grid_remove()
            else:
                scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S), padx=(5, 0))

            canvas.itemconfig(canvas_window, width=canvas_width, height=max(frame_height, canvas_height))
            canvas.configure(scrollregion=canvas.bbox('all'))

        def _on_mousewheel(event):
            if scrollbar.winfo_ismapped():
                canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')

        def _on_scroll(event):
            def precise_scroll_deltas(dxdy):
                deltaX = dxdy >> 16
                low = dxdy & 0xFFFF
                deltaY = low if low < 0x8000 else low - 0x10000
                return deltaX, deltaY

            if scrollbar.winfo_ismapped():
                _, delta_y = precise_scroll_deltas(event.delta)
                canvas.yview_scroll(int(-1 * (delta_y / 5)), 'units')

        def _on_enter(event):
            canvas.bind_all('<MouseWheel>', _on_mousewheel)
            if sys.platform == 'darwin':
                canvas.bind_all('<TouchpadScroll>', _on_scroll)

        def _on_leave(event):
            canvas.unbind_all('<MouseWheel>')
            if sys.platform == 'darwin':
                canvas.unbind_all('<TouchpadScroll>')

        canvas.bind('<Configure>', update_scrollable_frame_height)
        canvas.bind('<Enter>', _on_enter)
        canvas.bind('<Leave>', _on_leave)

    def _create_buttons(self, parent):
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=2, column=0, columnspan=2)

        ttk.Button(button_frame, text='Save', command=self.save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text='Reset to Defaults', command=self.reset_to_defaults).pack(side=tk.LEFT, padx=5)

    def _configure_grid_weights(self, main_frame):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

    def _create_option_widget(self, parent, option_data, row, category):
        if len(option_data) == 4:
            option, opt_type, help_text, dropdown_values = option_data
        else:
            option, opt_type, help_text = option_data
            dropdown_values = None

        if option == 'tray_icon' and self.is_bundled:
            return row

        if 'special_screen_capture' in opt_type:
            is_window_area = 'window_area' in opt_type
            return self._create_screen_capture_widget(parent, option, help_text, row, is_window_area)

        frame = self._create_option_frame(parent, row)

        ttk.Label(frame, text=f'{option}:').grid(row=0, column=0, sticky=tk.W, padx=(0, 10))

        input_frame = ttk.Frame(frame)
        input_frame.grid(row=0, column=1, sticky=(tk.W, tk.E))

        var, widget = self._create_widget_by_type(opt_type, input_frame, option, dropdown_values)

        ttk.Label(frame, text=help_text, style='Help.TLabel').grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(2, 0))

        self.widgets[option] = {
            'var': var,
            'widget': widget,
            'type': opt_type,
            'frame': frame
        }

        if option in ['read_from', 'read_from_secondary', 'write_to']:
            var.trace_add('write', lambda *_: self._update_general_state())

        self._add_option_buttons(input_frame, option, var, widget, category)

        frame.columnconfigure(1, weight=1)
        input_frame.columnconfigure(0, weight=1)

        return row + 1

    def _create_option_frame(self, parent, row):
        frame = ttk.Frame(parent, padding=5)
        frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=2)
        return frame

    def _create_widget_by_type(self, widget_type, parent, option, dropdown_values):
        widget_creators = {
            'bool': self._create_bool_widget,
            'int': self._create_int_widget,
            'float': self._create_float_widget,
            'dropdown': self._create_dropdown_widget,
            'multicheckbox': self._create_multicheckbox_widget,
            'str': self._create_string_widget
        }

        creator = widget_creators[widget_type]
        return creator(parent, option, dropdown_values)

    def _create_bool_widget(self, parent, option, dropdown_values):
        var = tk.BooleanVar()
        widget = ttk.Checkbutton(parent, variable=var)
        widget.pack(side=tk.LEFT, fill=tk.X)
        return var, widget

    def _create_int_widget(self, parent, option, dropdown_values):
        var = tk.StringVar()
        widget = ttk.Spinbox(parent, textvariable=var, from_=-2, to=1000)
        widget.pack(side=tk.LEFT, fill=tk.X)
        return var, widget

    def _create_float_widget(self, parent, option, dropdown_values):
        var = tk.StringVar()
        widget = ttk.Spinbox(parent, textvariable=var, from_=-1.0, to=1000.0, increment=0.1, format='%.1f')
        widget.pack(side=tk.LEFT, fill=tk.X)
        return var, widget

    def _create_dropdown_widget(self, parent, option, dropdown_values):
        var = tk.StringVar()

        if option == 'engine':
            dropdown_values = [''] + self.all_engines
        elif option == 'engine_secondary':
            dropdown_values = [''] + self.secondary_engines

        widget = ttk.Combobox(parent, textvariable=var, values=dropdown_values, state='readonly')
        widget.pack(side=tk.LEFT, fill=tk.X)
        return var, widget

    def _create_multicheckbox_widget(self, parent, option, dropdown_values):
        checkbox_frame = ttk.Frame(parent)
        checkbox_frame.pack(side=tk.LEFT, fill=tk.X)

        vars_dict = {}
        for i, display_name in enumerate(self.all_engines):
            var = tk.BooleanVar()
            cb = ttk.Checkbutton(checkbox_frame, text=display_name, variable=var)
            row_pos = i // 3
            col_pos = i % 3
            cb.grid(row=row_pos, column=col_pos, sticky=tk.W, padx=10, pady=2)
            vars_dict[display_name] = var

            var.trace_add('write', lambda *_: self._update_engine_state())

        return vars_dict, checkbox_frame

    def _create_string_widget(self, parent, option, dropdown_values):
        var = tk.StringVar()
        widget = ttk.Entry(parent, textvariable=var)
        widget.pack(side=tk.LEFT, fill=tk.X)
        return var, widget

    def _add_option_buttons(self, input_frame, option, var, widget, category):
        if option in ['read_from', 'read_from_secondary', 'write_to']:
            ttk.Button(input_frame, text='...', width=2, command=lambda: self._open_picker(option, var)).pack(side=tk.LEFT, padx=(5, 0))
            widget.pack(expand=True)

        if category == 'Hotkeys':
            ttk.Button(input_frame, text='...', width=2, command=lambda: self._start_hotkey_recording(option)).pack(side=tk.LEFT, padx=(5, 0))
            widget.pack(expand=True)

    def _create_screen_capture_widget(self, parent, option, help_text, row, is_window_area=False):
        frame = self._create_option_frame(parent, row)

        ttk.Label(frame, text=f'{option}:').grid(row=0, column=0, sticky=tk.W, padx=(0, 10))

        input_frame = ttk.Frame(frame)
        input_frame.grid(row=0, column=1, sticky=(tk.W, tk.E))

        if is_window_area:
            dropdown_values = ['automatic selection', 'coordinates', 'entire window']
            placeholder_texts = {'coordinates': 'x1,y1,x2,y2'}
        else:
            dropdown_values = ['automatic selection', 'coordinates', 'entire screen', 'window']
            placeholder_texts = {
                'coordinates': 'x1,y1,x2,y2',
                'entire screen': '1',
                'window': 'window name'
            }

        dropdown_var = tk.StringVar()
        dropdown = ttk.Combobox(input_frame, textvariable=dropdown_var, values=dropdown_values, state='readonly')
        dropdown.pack(side=tk.LEFT, fill=tk.X)

        text_var = tk.StringVar()
        textbox = ttk.Entry(input_frame, textvariable=text_var, width=20)

        def update_textbox_visibility():
            mode = dropdown_var.get()
            if mode in placeholder_texts:
                textbox.pack(side=tk.LEFT, fill=tk.X, padx=(5, 0))
                textbox.delete(0, tk.END)
                textbox.insert(0, placeholder_texts[mode])
            else:
                textbox.pack_forget()
                text_var.set('')

        dropdown_var.trace_add('write', lambda *_: update_textbox_visibility())

        ttk.Label(frame, text=help_text, style='Help.TLabel').grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(2, 0))

        widget_type = 'special_screen_capture_window_area' if is_window_area else 'special_screen_capture_area'
        self.widgets[option] = {
            'type': widget_type,
            'dropdown_var': dropdown_var,
            'text_var': text_var,
            'frame': frame
        }

        if not is_window_area:
            dropdown_var.trace_add('write', lambda *_: self._update_screen_capture_state())

        frame.columnconfigure(1, weight=1)
        input_frame.columnconfigure(0, weight=1)

        return row + 1

    def _add_engine_configuration_sections(self, parent, start_row):
        row = start_row

        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        row += 1

        self.engine_config_container = ttk.Frame(parent)
        self.engine_config_container.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E))
        self.engine_config_container.columnconfigure(1, weight=1)

        container_row = 0
        for config_entry, engine_names in self.config_entry_to_engines.items():
            self._create_engine_config_frame(config_entry, engine_names, container_row)
            container_row += 1

        return row + 1

    def _create_engine_config_frame(self, config_entry, engine_names, row):
        label_text = engine_names[0] if len(engine_names) == 1 else ', '.join(engine_names)

        frame = ttk.LabelFrame(self.engine_config_container, text=label_text, padding=10)
        frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 10), ipadx=5)
        frame.columnconfigure(1, weight=1)

        self.engine_config_widgets[config_entry] = {
            'frame': frame,
            'engine_names': engine_names,
            'widgets': {}
        }

        option_row = 0
        for option_data in self.engine_config_options[config_entry]:
            option_name, option_type, help_text, default_value = option_data
            self._create_engine_config_option(frame, config_entry, option_name, option_type, help_text, default_value, option_row)
            option_row += 1

        return frame

    def _create_engine_config_option(self, parent, config_entry, option_name, option_type, help_text, default_value, row):
        option_frame = ttk.Frame(parent, padding=2)
        option_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)

        ttk.Label(option_frame, text=f'{option_name}:').grid(row=0, column=0, sticky=tk.W, padx=(0, 10))

        input_frame = ttk.Frame(option_frame)
        input_frame.grid(row=0, column=1, sticky=(tk.W, tk.E))

        var, widget = self._create_engine_config_widget(option_type, input_frame, default_value)

        ttk.Label(option_frame, text=help_text, style='Help.TLabel').grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(2, 0))

        self.engine_config_widgets[config_entry]['widgets'][option_name] = {
            'var': var,
            'widget': widget,
            'type': option_type
        }

        option_frame.columnconfigure(1, weight=1)

    def _create_engine_config_widget(self, widget_type, parent, default_value):
        if widget_type == 'bool':
            var = tk.BooleanVar()
            widget = ttk.Checkbutton(parent, variable=var)
            var.set(str(default_value).lower() == 'true')
            widget.pack(side=tk.LEFT)
        elif widget_type == 'int':
            var = tk.StringVar()
            widget = ttk.Spinbox(parent, textvariable=var, from_=0, to=10000)
            var.set(str(default_value))
            widget.pack(side=tk.LEFT)
        else:
            var = tk.StringVar()
            widget = ttk.Entry(parent, textvariable=var)
            var.set(str(default_value))
            widget.pack(side=tk.LEFT, fill=tk.X, expand=True)

        return var, widget

    def _start_hotkey_recording(self, option):
        widget_info = self.widgets[option]

        def on_hotkey_recorded(hotkey_str):
            widget_info['var'].set(hotkey_str)

        hotkey_recorder.on_hotkey_recorded = on_hotkey_recorded
        hotkey_recorder.start_recording(self.root)

    def _open_picker(self, option, var):
        if option in ['read_from', 'read_from_secondary']:
            folder_path = filedialog.askdirectory(
                title=f'Select directory for {option}',
                mustexist=True
            )
            if folder_path:
                var.set(folder_path)
                self._update_general_state()
        elif option == 'write_to':
            file_path = filedialog.asksaveasfilename(
                title='Select output text file',
                defaultextension='.txt',
                filetypes=[('Text files', '*.txt'), ('All files', '*.*')]
            )
            if file_path:
                var.set(file_path)
                self._update_general_state()

    def _update_ui_state(self):
        self._update_screen_capture_state()
        self._update_general_state()
        self._update_engine_state()

    def _is_folder_file_selected(self, option):
        directory_inputs = ['clipboard', 'websocket', 'unixsocket', 'screencapture']

        option_val = self._get_widget_value(option)
        if option_val and option_val not in directory_inputs:
            return True

        return False

    def _update_screen_capture_state(self):
        screen_capture_area_info = self.widgets.get('screen_capture_area')
        dropdown_val = screen_capture_area_info['dropdown_var'].get()

        for option in ['screen_capture_window_area', 'screen_capture_only_active_windows']:
            widget_info = self.widgets.get(option)
            frame = widget_info.get('frame')
            frame.grid() if dropdown_val == 'window' else frame.grid_remove()

        self._update_tab_canvas_scrollregion('Screen capture')

    def _update_general_state(self):
        widget_info = self.widgets['websocket_port']
        frame = widget_info.get('frame')
        show_websocket = self._should_show_websocket_port()
        frame.grid() if show_websocket else frame.grid_remove()

        widget_info = self.widgets['delay_seconds']
        frame = widget_info.get('frame')
        show_delay = self._should_show_delay_seconds()
        frame.grid() if show_delay else frame.grid_remove()

        widget_info = self.widgets['delete_images']
        frame = widget_info.get('frame')
        show_delete = self._is_folder_file_selected('read_from') or self._is_folder_file_selected('read_from_secondary')
        frame.grid() if show_delete else frame.grid_remove()

        dropdown_options = ['read_from', 'read_from_secondary', 'write_to']
        for option in dropdown_options:
            widget_info = self.widgets[option]
            combobox = widget_info.get('widget')
            if self._is_folder_file_selected(option):
                current_value = widget_info['var'].get()
                if current_value:
                    adjusted_width = min(max(len(current_value) + 2, 18), 38)
                    combobox.configure(width=adjusted_width)
                else:
                    combobox.configure(width=18)
            else:
                combobox.configure(width=18)

        self._update_tab_canvas_scrollregion('General')

    def _update_engine_state(self):
        selected_engines = self._get_selected_engines()
        show_all = len(selected_engines) == 0

        if show_all:
            available_engines = self.all_engines
            available_secondary = self.secondary_engines
        else:
            available_engines = list(selected_engines)
            available_secondary = [engine for engine in selected_engines if engine in self.secondary_engines]

        self._update_dropdown('engine', available_engines)
        self._update_dropdown('engine_secondary', available_secondary)

        show_container = False
        for config_entry, config_data in self.engine_config_widgets.items():
            frame = config_data['frame']
            engine_names = config_data['engine_names']

            # Show if any engine in this group is selected OR no engines are selected
            should_show = show_all or any(engine_name in selected_engines for engine_name in engine_names)

            if should_show:
                show_container = True
                frame.grid()
            else:
                frame.grid_remove()

        if show_container:
            self.engine_config_container.grid()
        else:
            self.engine_config_container.grid_remove()

        self._update_tab_canvas_scrollregion('Engines')

    def _update_dropdown(self, option_name, available_values):
        widget_info = self.widgets[option_name]
        current_value = widget_info['var'].get()
        dropdown = widget_info['widget']

        new_values = [''] + available_values
        dropdown['values'] = new_values

        if current_value not in new_values:
            widget_info['var'].set('')

    def _should_show_websocket_port(self):
        websocket_options = ['read_from', 'read_from_secondary', 'write_to']
        for option in websocket_options:
            value = self._get_widget_value(option)
            if value == 'websocket':
                return True
        return False

    def _should_show_delay_seconds(self):
        non_delay_options = ['websocket', 'unixsocket', 'screencapture']
        read_from_val = self._get_widget_value('read_from')
        if read_from_val not in non_delay_options:
            return True
        read_from_secondary_val = self._get_widget_value('read_from_secondary')
        if read_from_secondary_val and read_from_secondary_val not in non_delay_options:
            return True
        return False

    def _update_tab_canvas_scrollregion(self, tab_name):
        canvas = self.tab_canvases[tab_name]
        scrollable_frame = self.tab_scrollable_frames[tab_name]
        scrollable_frame.update_idletasks()
        canvas.update_idletasks()
        frame_height = scrollable_frame.winfo_reqheight()
        canvas_height = canvas.winfo_height()
        canvas_width = canvas.winfo_width()

        scrollbar = self.tab_scrollbars[tab_name]
        if frame_height <= canvas_height:
            scrollbar.grid_remove()
        else:
            scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S), padx=(5, 0))

        canvas.itemconfig('scrollable_frame', width=canvas_width, height=max(frame_height, canvas_height))
        canvas.configure(scrollregion=canvas.bbox('all'))

    def _get_selected_engines(self):
        selected_engines = [engine_name for engine_name, engine_var in self.widgets['engines']['var'].items() if engine_var.get()]
        return selected_engines

    def _load_config(self):
        # Set all widgets to default values first
        self._set_all_defaults(update_ui=False)

        if not os.path.exists(self.config_path):
            messagebox.showwarning('File Not Found', f'Config file not found at:\n{self.config_path}\nUsing defaults.')
            self.status_var.set('Using default configuration')
            self._update_ui_state()
            return

        try:
            self.config = configparser.ConfigParser()
            self.config.read(self.config_path, encoding='utf-8')

            if 'general' in self.config:
                for option in self.widgets:
                    if option in self.config['general']:
                        value = self.config['general'][option].strip()
                        self._set_widget_value(option, value)
            self._load_engine_config()

            self.status_var.set(f'Configuration loaded from {self.config_path}')
            self._update_ui_state()
        except Exception as e:
            messagebox.showerror('Error', f'Failed to load config: {str(e)}')
            self.status_var.set('Error loading configuration')

    def _load_engine_config(self):
        for config_entry, config_data in self.engine_config_widgets.items():
            if config_entry in self.config:
                for option_name, widget_info in config_data['widgets'].items():
                    if option_name in self.config[config_entry]:
                        value = self.config[config_entry][option_name].strip()
                        self._set_engine_config_value(config_entry, option_name, value)

    def _set_all_defaults(self, update_ui=True):
        for option in self.widgets:
            default_value = self.default_values[option]
            self._set_widget_value(option, str(default_value))
        self._reset_engine_configs()

        if update_ui:
            self._update_ui_state()
            self.status_var.set('Reset to default values')

    def _set_widget_value(self, option, value):
        widget_info = self.widgets[option]
        value = self._strip_quotes(value)

        if widget_info['type'] == 'special_screen_capture_area':
            self._set_screen_capture_value(widget_info, value)
        elif widget_info['type'] == 'special_screen_capture_window_area':
            self._set_screen_capture_window_value(widget_info, value)
        elif widget_info['type'] == 'bool':
            var = widget_info['var']
            var.set(value.lower() == 'true')
        elif widget_info['type'] == 'int':
            var = widget_info['var']
            try:
                var.set(int(value))
            except ValueError:
                var.set(self.default_values.get(option, 0))
        elif widget_info['type'] == 'float':
            var = widget_info['var']
            try:
                var.set(float(value))
            except ValueError:
                var.set(self.default_values.get(option, 0.0))
        elif widget_info['type'] == 'multicheckbox':
            self._set_multicheckbox_value(widget_info, value)
        elif option in ['engine', 'engine_secondary']:
            var = widget_info['var']
            if value in self.engine_class_to_display:
                var.set(self.engine_class_to_display[value])
            else:
                var.set(value)
        else:
            var = widget_info['var']
            var.set(value)

    def _strip_quotes(self, value):
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            return value[1:-1]
        return value

    def _set_screen_capture_value(self, widget_info, value):
        if value == '':
            widget_info['dropdown_var'].set('automatic selection')
            widget_info['text_var'].set('')
        elif value.startswith('screen_'):
            widget_info['dropdown_var'].set('entire screen')
            try:
                screen = int(value.replace('screen_', ''))
            except ValueError:
                screen = 1
            if screen < 1:
                screen = 1
            widget_info['text_var'].set(str(screen))
        elif len(value.replace('_', ',').split(',')) % 4 == 0:
            widget_info['dropdown_var'].set('coordinates')
            widget_info['text_var'].set(value)
        else:
            widget_info['dropdown_var'].set('window')
            widget_info['text_var'].set(value)

    def _set_screen_capture_window_value(self, widget_info, value):
        if value == '':
            widget_info['dropdown_var'].set('automatic selection')
            widget_info['text_var'].set('')
        elif len(value.replace('_', ',').split(',')) % 4 == 0:
            widget_info['dropdown_var'].set('coordinates')
            widget_info['text_var'].set(value)
        else:
            widget_info['dropdown_var'].set('entire window')
            widget_info['text_var'].set('')

    def _set_multicheckbox_value(self, widget_info, value):
        var = widget_info['var']
        enabled_engines = [e.strip() for e in value.split(',') if e.strip()]
        enabled_display_names = []

        for internal_name in enabled_engines:
            display_name = self.engine_class_to_display.get(internal_name, internal_name)
            enabled_display_names.append(display_name)

        for engine, engine_var in var.items():
            engine_var.set(engine in enabled_display_names)

    def _set_engine_config_value(self, config_entry, option, value):
        if (config_entry in self.engine_config_widgets and

            option in self.engine_config_widgets[config_entry]['widgets']):

            widget_info = self.engine_config_widgets[config_entry]['widgets'][option]
            var = widget_info['var']

            value = self._strip_quotes(value)

            if widget_info['type'] == 'bool':
                var.set(value.lower() == 'true')
            else:
                var.set(value)

    def _get_widget_value(self, option):
        widget_info = self.widgets[option]

        if widget_info['type'] == 'special_screen_capture_area':
            return self._get_screen_capture_value(widget_info)
        elif widget_info['type'] == 'special_screen_capture_window_area':
            return self._get_screen_capture_window_value(widget_info)
        elif widget_info['type'] == 'bool':
            return str(widget_info['var'].get())
        elif widget_info['type'] in ['int', 'float']:
            return str(widget_info['var'].get())
        elif widget_info['type'] == 'multicheckbox':
            return self._get_multicheckbox_value(widget_info)
        elif option in ['engine', 'engine_secondary']:
            return self._get_engine_dropdown_value(widget_info)
        else:
            return self._get_string_value(widget_info, option)

    def _get_screen_capture_value(self, widget_info):
        dropdown_val = widget_info['dropdown_var'].get()
        text_val = widget_info['text_var'].get().strip()

        if dropdown_val == 'coordinates':
            return text_val
        elif dropdown_val == 'entire screen':
            try:
                screen = int(text_val)
            except ValueError:
                screen = 1
            if screen < 1:
                screen = 1
            return f'screen_{screen}'
        elif dropdown_val == 'window':
            return text_val
        else:
            return ''

    def _get_screen_capture_window_value(self, widget_info):
        dropdown_val = widget_info['dropdown_var'].get()
        text_val = widget_info['text_var'].get().strip()

        if dropdown_val == 'automatic selection':
            return ''
        elif dropdown_val == 'coordinates':
            return text_val
        else:
            return 'window'

    def _get_multicheckbox_value(self, widget_info):
        var = widget_info['var']
        selected_display_names = [engine for engine, engine_var in var.items() if engine_var.get()]
        selected_internal_names = []

        for display_name in selected_display_names:
            engine_class = self.engine_name_to_class.get(display_name)
            selected_internal_names.append(engine_class.name)

        return ','.join(selected_internal_names)

    def _get_engine_dropdown_value(self, widget_info):
        display_name = widget_info['var'].get()
        if not display_name:
            return ''
        engine_class = self.engine_name_to_class.get(display_name)
        return engine_class.name

    def _get_string_value(self, widget_info, option):
        value = widget_info['var'].get()

        if any(char in value for char in [' ', '\\', '\t']):
            return f'"{value}"'

        return value

    def _get_engine_config_value(self, config_entry, option):
        widget_info = self.engine_config_widgets[config_entry]['widgets'][option]
        value = widget_info['var'].get()

        if widget_info['type'] == 'bool':
            return str(value)
        else:
            if value and any(char in value for char in [' ', '\\', '\t']):
                return f'"{value}"'
            return value

    def save_config(self):
        try:
            config_dir = os.path.dirname(self.config_path)
            if not os.path.exists(config_dir):
                os.makedirs(config_dir)

            config = configparser.ConfigParser()
            config['general'] = {}

            for option in self.widgets:
                value = self._get_widget_value(option)
                config['general'][option] = value
            self._save_engine_config(config)

            with open(self.config_path, 'w', encoding='utf-8') as f:
                config.write(f)

            self.status_var.set(f'Configuration saved to {self.config_path}')
            messagebox.showinfo('Success', 'Configuration saved successfully!')
        except Exception as e:
            messagebox.showerror('Error', f'Failed to save config: {str(e)}')

    def _save_engine_config(self, config):
        for config_entry, config_data in self.engine_config_widgets.items():
            if config_data['widgets']:
                if config_entry not in config:
                    config[config_entry] = {}

                for option_name, widget_info in config_data['widgets'].items():
                    value = self._get_engine_config_value(config_entry, option_name)
                    config[config_entry][option_name] = value

    def _reset_engine_configs(self):
        for config_entry, config_data in self.engine_config_widgets.items():
            for option_name, option_type, help_text, default_value in self.engine_config_options[config_entry]:
                if option_name in config_data['widgets']:
                    self._set_engine_config_value(config_entry, option_name, str(default_value))

    def reset_to_defaults(self):
        self._set_all_defaults(update_ui=True)


def main():
    if not editor_available:
        print('tkinter is not installed, unable to open editor')
        sys.exit(1)

    global hotkey_recorder
    hotkey_recorder = HotkeyRecorder()
    hotkey_recorder.start_listener()

    if sys.platform == 'win32':
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    root = tk.Tk()
    app = ConfigGUI(root)
    root.mainloop()
    root.update()

    hotkey_recorder.stop_listener()
