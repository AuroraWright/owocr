<div align="center">
<img alt="" src="https://raw.githubusercontent.com/AuroraWright/owocr/refs/heads/master/owocr/data/icon.png" height="128" style="border-radius: 100%;">
<h1>OwOCR</h1>
</div>

OwOCR is a text recognition tool that continuously scans for images and performs OCR (Optical Character Recognition) on them. Its main focus is Japanese, but it works for many other languages.

## Demo

### Visual novel

https://github.com/user-attachments/assets/f2196b23-f2e7-4521-820a-88c8bedb9d8e

### Manga

https://github.com/user-attachments/assets/f061854d-d20f-43e8-8c96-af5a0ea26f43

## Installation/basic usage (easy Windows/macOS packages)

Easy to install Windows and macOS packages can be downloaded [here](https://github.com/AuroraWright/owocr/releases).
- On Windows, just extract the zip anywhere and double click on "owocr".
- On macOS, just double click on the dmg and drag the owocr app to the Application folder, like most macOS apps. You will be asked to grant two permissions to owocr the first time it starts (Accessibility and Screen Capture), just follow the prompts to do so.
- A "Log Viewer" window will shop up, showing information messages. A tray icon in the macOS menu bar/Windows task bar will show up. You can close the log viewer if you want.
- By default owocr monitors the clipboard for images and outputs recognized text back to the clipboard. You can change this from the configuration, accessible from the tray icon.
- With a left click on the tray icon you can pause/unpause on Windows, from the right click menu (left click on macOS) you can change the engine, pause/unpause, change the screen capture area selection, take a screenshot of the selected screen/window, launch the configuration, and reopen the log viewer if you closed it. The icon will be dimmed to show when owocr is paused.
- In these versions all the OCR engines and features are already available, you don't need to install anything else. The tray icon is always enabled and can't be turned off.

## Installation (terminal+Python, all operating systems)

OwOCR has been tested on Python 3.11, 3.12 and 3.13. It can be installed with `pip install owocr` after you install Python. You also need to have one or more OCR engines, check the list below for instructions. I recommend installing at least Google Lens on any operating system, and OneOCR if you are on Windows. Bing is pre-installed, Apple Vision and Live Text come pre-installed on macOS.

## Usage (terminal)

```shell
owocr
```

This default behavior monitors the clipboard for images and outputs recognized text back to the clipboard.

```shell
owocr_config
```

This opens the interface where you can change all the options.

From the terminal window you can pause/unpause with `p` or terminate with `t`/`q`, switch between engines with `s` or the engine-specific keys (from the engine list below).\
The tray icon can also be used as explained above.\
All command-line options and their descriptions can be viewed with: `owocr -h`.

## Main features

- Multiple input sources: clipboard, folders, websockets, unix domain socket, screen capture, OBS
- Multiple output destinations: clipboard, text files, and websockets
- Integrates well with Windows, macOS and Linux, supporting operating system features like notifications and a tray icon
- Capture from specific screen areas, windows, of areas within windows (window capture is only supported on Windows/macOS/Wayland). This also tries to capture entire sentences and filter all repetitions. If you use an online engine like Lens I recommend setting a secondary local engine (OneOCR on Windows, Apple Live Text on macOS and meikiocr on Linux). With this "two pass" system only the changed areas are sent to the online service, allowing for both speed and accuracy
- Control from the tray icon or the terminal window
- Control from anywhere through keyboard shortcuts: you can set hotkeys for pausing, switching engines, taking a screenshot of the selected screen/window and changing the screen capture area selection
- Read from a unix domain socket `/tmp/owocr.sock` on macOS/Linux
- Furigana filter, works by default with Japanese text (both vertical and horizontal)

## Manual configuration

The configuration file is stored in `~/.config/owocr_config.ini` on Linux/macOS, or `C:\Users\yourusername\.config\owocr_config.ini` on Windows.\
A sample config file is available at: [owocr_config.ini](https://raw.githubusercontent.com/AuroraWright/owocr/master/owocr_config.ini)

## Notes about Linux support

While I've done all I could to support Linux (specifically Wayland), not everything might work with all setups. Specifically:

- There are two ways of reading images from and writing text to the clipboard on Wayland. One requires a compositor which supports the extension "ext-data-control" and this should work out of the box with owocr by default. [ext_data_control compatibility chart](https://wayland.app/protocols/wayland-protocols/336#compositor-support) (worth noting GNOME/Mutter doesn't support it, but e.g. KDE/KWin does).\
The alternative is through `wl-clipboard` (preinstalled in most distributions), but this will try to steal your focus constantly (due to Wayland's security design), limiting usability.\
To switch to wl-clipboard, enable `wayland_use_wlclipboard` in `owocr_config` -> Advanced.
- Reading from screen capture works on Wayland. The way it's designed is that your monitor/monitor selection/window selection in the operating system popup counts as a "virtual screen" to owocr.\
By default the automatic coordinate selector will be launched to select one/more areas, as explained above.\
Using "whole screen" 1 in the configuration/`owocr -r=screencapture -sa=screen_1` will use the whole selection.\
Using manual window names is not supported and will be ignored.
- Keyboard combos/keyboard inputs in the coordinate selector might not work on Wayland. From my own testing they work on KDE (if you enable keyboard access in "Legacy X11 App Support" under "Application Permissions") but not GNOME. A workaround involves running pynput with the uinput backend, but this requires exposing your input devices (they will be accessible without root):\
`sudo chmod u+s $(which dumpkeys)`\
`sudo usermod -a -G $(stat -c %G /dev/input/event0) $(whoami)`\
Then launch owocr with: `PYNPUT_BACKEND_KEYBOARD=uinput owocr -r screencapture` or add `PYNPUT_BACKEND_KEYBOARD=uinput` to your environment variables.
- The tray icon requires installing [this extension](https://extensions.gnome.org/extension/615/appindicator-support) on GNOME (works out of the box on KDE)
- X11 partially works but uses more resources for scanning the clipboard and doesn't support window capturing at all (only screens/screen selections).

## Supported engines

### Local
- Chrome Screen AI - **Recommended** - Possibly the best local engine to date. Setup will be done automatically, but you can also manually download the zip for your operating system from [this link](https://chrome-infra-packages.appspot.com/p/chromium/third_party/screen-ai) and extract it to the `C:/Users/yourusername/.config/screen_ai` folder (Windows) or the `~/.config/screen_ai` folder (macOS/Linux). → Terminal: install with `pip install "owocr[screenai]"`, key: `j`
- OneOCR - **Windows 10/11 only - Recommended** - One of the best local engines to date. On Windows 11, setup will be done automatically if you have Snipping Tool from the Microsoft Store (it's installed by default). On Windows 10 you need to copy 3 system files from Windows 11 to use it, refer to the readme [here](https://github.com/AuroraWright/oneocr). It can also be used by installing oneocr on a Windows virtual machine and running the server there (`oneocr_serve`) and specifying the IP address of the Windows VM/machine in the config file. → Terminal: install with `pip install "owocr[oneocr]"`, key: `z`
- Apple Live Text (VisionKit framework) - **macOS only - Recommended** - One of the best local engines to date. It should be the same as Vision except that in Sonoma Apple added vertical text reading. → Terminal key: `d`
- Apple Vision framework - **macOS only** - Older version of Live Text. → Terminal key: `a`
- [meikiocr](https://github.com/rtr46/meikiocr) - **Recommended** - Comparable to OneOCR in accuracy and CPU latency, best local option for Linux users. Can't process vertical text and is limited to 64 text lines and 48 characters per line. → Terminal: install with `pip install "owocr[meikiocr]"`, if you have a Nvidia GPU you can do `pip uninstall onnxruntime && pip install onnxruntime-gpu` which makes it the fastest OCR available. Key: `k`
- [Manga OCR](https://github.com/kha-white/manga-ocr) (with optional [comic-text-detector](https://github.com/dmMaze/comic-text-detector) as segmenter) → Terminal: install with `pip install "owocr[mangaocr]"`, keys: `m` (regular, ideal for small text areas), `n` (segmented, ideal for manga panels/larger images with multiple text areas)
- WinRT OCR: **Windows 10/11 only** - It can also be used by installing winocr on a Windows virtual machine and running the server there (`winocr_serve`) and specifying the IP address of the Windows VM/machine in the config file. → Terminal: install with `pip install "owocr[winocr]"`, key: `w`
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) → Terminal: install with `pip install "owocr[easyocr]"`, key: `e`
- [RapidOCR](https://github.com/RapidAI/RapidOCR) → Terminal: install with `pip install "owocr[rapidocr]"`, key: `r`

### Cloud
- Google Lens - **Recommended** - Arguably the best OCR engine to date. → Terminal: install with `pip install "owocr[lens]"`, key: `l`
- Bing - **Recommended** - Close second best. → Terminal key: `b`
- Google Vision - You need a service account .json file named google_vision.json in `user directory/.config/` → Terminal: install with `pip install "owocr[gvision]"`, key: `g`
- Azure Image Analysis - You need to specify an api key and an endpoint in the config file → Terminal: install with `pip install "owocr[azure]"`, key: `v`
- OCRSpace - You need to specify an api key in the config file. → Terminal key: `o`

## Links

<a href="https://pypi.org/project/owocr">
	<img alt="Available on PyPI" title="Available on PyPI" src="https://img.shields.io/pypi/v/owocr?label=pypi&color=ffd242">
</a>

## Acknowledgments

This uses code from/references these people/projects:
- Viola for working on the Google Lens implementation (twice!) and helping with the pyobjc VisionKit code!
- @rtr46 for contributing a big overhaul allowing for coordinate support and JSON output, and for the initial Chrome Screen AI implementation!
- @bpwhelan for contributing code for other language support and for his ideas (like two pass processing) originally implemented in the Game Sentence Miner fork of owocr
- @bropines for the Bing code ([Github issue](https://github.com/AuroraWright/owocr/issues/10))
- @ronaldoussoren for helping with the pyobjc VisionKit code
- [Manga OCR](https://github.com/kha-white/manga-ocr) for inspiring and being the project owocr was originally derived from
- [Mokuro](https://github.com/kha-white/mokuro) for the comic text detector integration code
- [ocrmac](https://github.com/straussmaximilian/ocrmac) for the Apple Vision framework API
- [ccylin2000_lipboard_monitor](https://github.com/vaimalaviya1233/ccylin2000_lipboard_monitor) for the Windows clipboard polling code
- vicky for the demo videos in this readme!
- nao for the awesome icon!
- [Steffo](https://github.com/Steffo99) for all his help in automating packaging/distribution with Github Actions!
- [kamperemu](https://github.com/kamperemu) for the OBS source implementation!
