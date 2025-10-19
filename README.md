# OwOCR

OwOCR is a command-line text recognition tool that continuously scans for images and performs OCR (Optical Character Recognition) on them. Its main focus is Japanese, but it works for many other languages.

## Installation

OwOCR has been tested on Python 3.11, 3.12 and 3.13. It can be installed with `pip install owocr` after you install Python. You also need to have one or more OCR engines, check the list below for instructions. I recommend installing at least Google Lens on any operating system, and OneOCR if you are on Windows. Bing is pre-installed, Apple Vision and Live Text come pre-installed on macOS.

## Basic usage

```
owocr
```

This default behavior monitors the clipboard for images and outputs recognized text back to the clipboard.

## Main features

- Multiple input sources: clipboard, folders, websockets, unix domain socket, and screen capture
- Multiple output destinations: clipboard, text files, and websockets
- Pause/unpause with `p` or terminate with `t`/`q` in the terminal, switch between engines with `s` or the engine-specific keys (from the engine list below)
- Capture from specific screen areas, windows, of areas within windows (window capture is only supported on Windows/macOS). This also tries to capture entire sentences and filter all repetitions. If you use an online engine like Lens I recommend setting a secondary local engine with the `-es` option: `-es=oneocr` on Windows and `-es=alivetext` on macOS. With this "two pass" system only the changed areas are sent to the online service, allowing for both speed and accuracy
- Multiple configurable keyboard combinations to control owocr from anywhere, including pausing, switching engines, taking a screenshot of the selected screen/window and running the automatic tool to re-select an area of the screen/window via drag and drop
- Read from a unix domain socket `/tmp/owocr.sock` on macOS/Linux
- Furigana filter, works by default with Japanese text (both vertical and horizontal)

## Common option examples

- Write text to a file: `owocr -w=<txt file path>`
- Read images from a folder: `owocr -r=<folder path>`
- Write text to a websocket: `owocr -w=websocket`
- Read from the screen or a portion of the screen (opens the automatic drag and drop selector): `owocr -r=screencapture`
- Read from a window having "Notepad" in the title: `owocr -r=screencapture -sa=Notepad`
- Read from a portion of a window having "Notepad" in the title (opens the automatic drag and drop selector): `owocr -r=screencapture -sa=Notepad -swa`

## Configuration

There are many more options and customization features. For complete documentation of all available settings:

- View all command-line options and their descriptions: `owocr -h`
- Check the automatically generated config file at `~/.config/owocr_config.ini` on Linux/macOS, or `C:\Users\yourusername\.config\owocr_config.ini` on Windows
- See a sample config file: [owocr_config.ini](https://raw.githubusercontent.com/AuroraWright/owocr/master/owocr_config.ini)

The command-line options/config file allow you to configure OCR providers, hotkeys, screen capture settings, notifications, and much more.

# Supported engines

## Local
- [Manga OCR](https://github.com/kha-white/manga-ocr) - install: `pip install owocr[mangaocr]` → key: `m`
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - install: `pip install owocr[easyocr]` → key: `e`
- [RapidOCR](https://github.com/RapidAI/RapidOCR) - install: `pip install owocr[rapidocr]` → key: `r`
- Apple Vision framework - Probably the best local engine to date. **macOS only - Recommended (pre-installed)** → key: `a`
- Apple Live Text (VisionKit framework) - It should be the same as Vision except that in Sonoma Apple added vertical text reading. **macOS only - Recommended (pre-installed)** → key: `d`
- WinRT OCR: install: `pip install owocr[winocr]`. It can also be used by installing winocr on a Windows virtual machine and running the server there (`winocr_serve`) and specifying the IP address of the Windows VM/machine in the config file. **Windows 10/11 only** → key: `w`
- OneOCR - install: `pip install owocr[oneocr]`. Close second local best to the Apple one. You need to copy 3 system files from Windows 11 to use it, refer to the readme [here](https://github.com/AuroraWright/oneocr). It can also be used by installing oneocr on a Windows virtual machine and running the server there (`oneocr_serve`) and specifying the IP address of the Windows VM/machine in the config file. **Windows 10/11 only - Recommended** → key: `z`

## Cloud
- Google Lens - install: `pip install owocr[lens]`. Arguably the best OCR engine to date. **Recommended** → key: `l`
- Bing - Close second best. **Recommended (pre-installed)** → key: `b`
- Google Vision: install: `pip install owocr[gvision]`, you also need a service account .json file named google_vision.json in `user directory/.config/` → key: `g`
- Azure Image Analysis: install: `pip install owocr[azure]`, you also need to specify an api key and an endpoint in the config file → key: `v`
- OCRSpace: you need to specify an api key in the config file → key: `o`

# Acknowledgments

This uses code from/references these people/projects:
- Viola for working on the Google Lens implementation (twice!) and helping with the pyobjc VisionKit code!
- @rtr46 for contributing a big overhaul allowing for coordinate support and JSON output
- @bpwhelan for contributing code for other language support and for his ideas (like two pass processing) originally implemented in the Game Sentence Miner fork of owocr
- @bropines for the Bing code ([Github issue](https://github.com/AuroraWright/owocr/issues/10))
- @ronaldoussoren for helping with the pyobjc VisionKit code
- [Manga OCR](https://github.com/kha-white/manga-ocr) for inspiring and being the project owocr was originally derived from
- [ocrmac](https://github.com/straussmaximilian/ocrmac) for the Apple Vision framework API
- [NadeOCR](https://github.com/Natsume-197/NadeOCR) for the Google Vision API
- [ccylin2000_lipboard_monitor](https://github.com/vaimalaviya1233/ccylin2000_lipboard_monitor) for the Windows clipboard polling code