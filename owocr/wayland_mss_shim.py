import re
import threading
import queue
import time

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

import dbus
from dbus.mainloop.glib import DBusGMainLoop

import mss as real_mss
from mss.exception import ScreenShotError
from mss.screenshot import ScreenShot, Size
from mss.models import Monitor


screencast = None
screencast_lock = threading.Lock()

class ScreenCastManager:
    def __init__(self):
        self.screen_cast_iface = 'org.freedesktop.portal.ScreenCast'
        self.frame_lock = threading.Lock()
        self.selected_event = threading.Event()
        self.ready_event = threading.Event()
        self.request_token_counter = 0
        self.session_token_counter = 0
        self.start()

    def __del__(self):
        self.stop()

    def _new_request_path(self):
        self.request_token_counter += 1
        token = f'u{self.request_token_counter}'
        path = f'/org/freedesktop/portal/desktop/request/{self.sender_name}/{token}'
        return path, token

    def _new_session_path(self):
        self.session_token_counter += 1
        token = f'u{self.session_token_counter}'
        path = f'/org/freedesktop/portal/desktop/session/{self.sender_name}/{token}'
        return path, token

    def _screen_cast_call(self, method, callback, *args, options=None):
        if options is None:
            options = {}
        request_path, request_token = self._new_request_path()

        self.bus.add_signal_receiver(
            callback,
            'Response',
            'org.freedesktop.portal.Request',
            'org.freedesktop.portal.Desktop',
            request_path
        )

        options['handle_token'] = request_token
        method(*(args + (options,)), dbus_interface=self.screen_cast_iface)

    def _on_session_closed(self, *args, **kwargs):
        self.stop()

    def _on_gst_message(self, bus, message):
        t = message.type
        if t in (Gst.MessageType.EOS, Gst.MessageType.ERROR):
            self.stop()

    def _process_sample(self, sample):
        buffer = sample.get_buffer()
        caps = sample.get_caps()
        caps_struct = caps.get_structure(0)
        width = caps_struct.get_value('width')
        height = caps_struct.get_value('height')

        success, map_info = buffer.map(Gst.MapFlags.READ)
        if success:
            try:
                data = bytes(map_info.data)
                return data, width, height
            finally:
                buffer.unmap(map_info)
        return None, width, height

    def _on_new_sample(self, appsink):
        try:
            sample = appsink.emit('pull-sample')
            assert sample is not None
            frame_data = self._process_sample(sample)
            assert frame_data[0] is not None
            with self.frame_lock:
                self.last_frame = frame_data
            self.ready_event.set()
        except:
            self.stop()
            return Gst.FlowReturn.Error
        return Gst.FlowReturn.OK

    def _play_pipewire_stream(self, node_id):
        portal = self.bus.get_object(
            'org.freedesktop.portal.Desktop',
            '/org/freedesktop/portal/desktop'
        )

        empty_dict = dbus.Dictionary(signature='sv')
        fd_object = portal.OpenPipeWireRemote(
            self.session,
            empty_dict,
            dbus_interface=self.screen_cast_iface
        )
        fd = fd_object.take()

        pipeline_str = (
            f'pipewiresrc fd={fd} path={node_id} ! '
            'video/x-raw,format={BGRA,BGRx} ! '
            'appsink name=sink max-buffers=1 drop=true emit-signals=true enable-last-sample=false qos=false sync=false'
        )
        self.pipeline = Gst.parse_launch(pipeline_str)
        bus = self.pipeline.get_bus()
        bus.connect('message', self._on_gst_message)
        appsink = self.pipeline.get_by_name('sink')
        appsink.connect('new-sample', self._on_new_sample)

        self.pipeline.set_state(Gst.State.PLAYING)

    def _on_start_response(self, response, results):
        if response != 0:
            self.stop()
            raise ScreenShotError(f'Failed to start screencast: {response}')

        self.selected_event.set()

        if results['streams']:
            node_id, stream_properties = results['streams'][0]
            self._play_pipewire_stream(node_id)
        else:
            self.stop()
            raise ScreenShotError('No streams available')

    def _on_select_sources_response(self, response, results):
        if response != 0:
            self.stop()
            raise ScreenShotError(f'Failed to select sources: {response}')

        portal = self.bus.get_object(
            'org.freedesktop.portal.Desktop',
            '/org/freedesktop/portal/desktop'
        )

        self._screen_cast_call(
            portal.Start,
            self._on_start_response,
            self.session,
            '',
            options={'multiple': False, 'types': dbus.UInt32(1 | 2), 'framerate': dbus.UInt32(30)}
        )

    def _on_create_session_response(self, response, results):
        if response != 0:
            self.stop()
            raise ScreenShotError(f'Failed to create session: {response}')

        self.session = results['session_handle']

        self.session_closed_signal = self.bus.add_signal_receiver(
            self._on_session_closed,
            'Closed',
            'org.freedesktop.portal.Session',
            'org.freedesktop.portal.Desktop',
            self.session
        )

        portal = self.bus.get_object(
            'org.freedesktop.portal.Desktop',
            '/org/freedesktop/portal/desktop'
        )

        self._screen_cast_call(
            portal.SelectSources,
            self._on_select_sources_response,
            self.session,
            options={'multiple': False, 'types': dbus.UInt32(1 | 2), 'framerate': dbus.UInt32(30)}
        )

    def _initialize_screencast(self):
        Gst.init(None)
        DBusGMainLoop(set_as_default=True)
        self.bus = dbus.SessionBus()
        self.sender_name = re.sub(r'\.', r'_', self.bus.get_unique_name()[1:])

        try:
            self.loop = GLib.MainLoop()

            portal = self.bus.get_object(
                'org.freedesktop.portal.Desktop',
                '/org/freedesktop/portal/desktop'
            )

            session_path, session_token = self._new_session_path()

            self._screen_cast_call(
                portal.CreateSession,
                self._on_create_session_response,
                options={'session_handle_token': session_token}
            )

            self.loop.run()
        except Exception as e:
            self.stop()
            raise ScreenShotError(f'Error initializing screencast: {e}')

    def request_frame(self):
        if self.ready_event.is_set():
            with self.frame_lock:
                if self.last_frame:
                    return self.last_frame
        return (None, 0, 0)

    def start(self):
        self.pipeline = None
        self.loop = None
        self.session = None
        self.last_frame = None
        self.selected_event.clear()

        self.init_thread = threading.Thread(target=self._initialize_screencast, daemon=True)
        self.init_thread.start()

    def stop(self):
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)

        if self.loop:
            self.loop.quit()

        self.selected_event.set()
        self.ready_event.clear()

class MSSWaylandShim:
    def __init__(self):
        global screencast
        with screencast_lock:
            if not screencast:
                screencast = ScreenCastManager()
                if not screencast.selected_event.wait(timeout=60):
                    raise ScreenShotError('Source selection timed out')
                if not screencast.ready_event.wait(timeout=3):
                    raise ScreenShotError('Screencast initialization timed out')
                time.sleep(1)
        self._create_monitors()

    @property
    def monitors(self):
        return self._monitors

    def grab(self, sct_params):
        frame_data = self._grab_screenshot(sct_params)
        bgra_data, crop_width, crop_height = frame_data

        return ScreenShot(bgra_data, self._monitors[0], size=Size(crop_width, crop_height))

    def _create_monitors(self):
        self._monitors = []

        frame = screencast.request_frame()
        if frame[0] is None:
            raise ScreenShotError('Invalid frame received')

        _, width, height = frame

        fake_monitor = Monitor({
            'top': 0,
            'left': 0,
            'width': width,
            'height': height
        })

        self._monitors.append(fake_monitor)
        self._monitors.append(fake_monitor)

    def _grab_screenshot(self, sct_params):
        frame = screencast.request_frame()
        if frame[0] is None:
            raise ScreenShotError('Invalid frame received')

        bgra_data, full_width, full_height = frame

        if sct_params != self._monitors[0]:
            crop_top = sct_params['top']
            crop_left = sct_params['left']
            crop_width = sct_params['width']
            crop_height = sct_params['height']

            crop_right = crop_left + crop_width
            crop_bottom = crop_top + crop_height

            crop_left = max(0, min(crop_left, full_width - 1))
            crop_top = max(0, min(crop_top, full_height - 1))
            crop_right = max(crop_left + 1, min(crop_right, full_width))
            crop_bottom = max(crop_top + 1, min(crop_bottom, full_height))

            if crop_right > crop_left and crop_bottom > crop_top:
                final_crop_width = crop_right - crop_left
                final_crop_height = crop_bottom - crop_top
                stride = full_width * 4

                cropped_data = bytearray(final_crop_width * final_crop_height * 4)

                for y in range(final_crop_height):
                    src_y = crop_top + y
                    src_start = src_y * stride + crop_left * 4
                    src_end = src_start + final_crop_width * 4
                    dst_start = y * final_crop_width * 4
                    cropped_data[dst_start:dst_start + (src_end - src_start)] = bgra_data[src_start:src_end]

                return cropped_data, final_crop_width, final_crop_height

        return bgra_data, full_width, full_height

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class MSSModuleShim:
    def mss(self):
        return MSSWaylandShim()

    def __getattr__(self, name):
        return getattr(real_mss, name)
