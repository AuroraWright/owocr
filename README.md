# OwOCR

Command line client for several Japanese OCR providers derived from [Manga OCR](https://github.com/kha-white/manga-ocr).

# Installation

This has been tested with Python 3.11. Newer/older versions might work. It can be installed with `pip install owocr`

# Supported providers

## Local providers
- [Manga OCR](https://github.com/kha-white/manga-ocr): refer to the readme for installation ("m" key)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR): refer to the readme for installation ("e" key)
- [RapidOCR](https://github.com/RapidAI/RapidOCR): refer to the readme for installation ("r" key)
- Apple Vision framework: this will work on macOS Ventura or later. In my experience, the best of the local providers for horizontal text ("a" key)
- WinRT OCR: this will work on Windows 10 or later if winocr (`pip install winocr`) is installed. It can also be used by installing winocr on a Windows virtual machine and running the server (`winocr_serve`), installing requests (`pip install requests`) and specifying the IP address of the Windows VM/machine in the config file (see below) ("w" key)

## Cloud providers
- Google Lens: Google Vision in disguise (no need for API keys!), however it needs to download a couple megabytes of data for each request. You need to install pyjson5 and requests (`pip install pyjson5 requests`) ("l" key)
- Google Vision: you need a service account .json file named google_vision.json in `user directory/.config/` and installing google-cloud-vision (`pip install google-cloud-vision`) ("g" key)
- Azure Image Analysis: you need to specify an api key and an endpoint in the config file (see below) and to install azure-ai-vision-imageanalysis (`pip install azure-ai-vision-imageanalysis`) ("v" key)

# Usage

It mostly functions like Manga OCR: https://github.com/kha-white/manga-ocr?tab=readme-ov-file#running-in-the-background
However:
- it supports reading images and/or writing text to a websocket when the -r=websocket and/or -w=websocket parameters are specified (port 7331 by default, configurable in the config file)
- it supports capturing the screen directly with -r screencapture. It will default to the entire first screen every 3 seconds, but a different screen/coordinates/window/delay can be specified in the config file. Instead of using a delay it's also possible to specify a keyboard combo (refer to the config file or the help page)
- you can pause/unpause the image processing by pressing "p" or terminate the script with "t" or "q" in the terminal window
- you can switch OCR provider pressing its corresponding keyboard key in the terminal window (refer to the list of keys above). You can also start the script paused with the -p option or with a specific provider with the -e option (refer to `owocr -h` for the list)
- holding ctrl or cmd at any time will pause image processing temporarily, or you can specify keyboard combos in the config file to pause/unpause and switch the OCR provider (refer to the config file or the help page)
- for systems where text can be copied to the clipboard at the same time as images, if `*ocr_ignore*` is copied with an image, the image will be ignored
- optionally, notifications can be enabled in the config file to show the text with a native OS notification
- optionally, you can speed up the online providers by installing fpng-py: `pip install fpng-py` (requires a developer environment on some operating systems/Python versions)
- optionally, you can improve filtering of non-Japanese text for screen capture by installing transformers: `pip install transformers`
- idle resource usage on macOS and Windows when reading from the clipboard has been eliminated using native OS polling
- a config file (to be created in `user directory/.config/owocr_config.ini`, on Windows `user directory` is the `C:\Users\yourusername` folder) can be used to configure the script, as an example to limit providers (to reduce clutter/memory usage) as well as specifying provider settings such as api keys etc. A sample config file is provided [here](https://raw.githubusercontent.com/AuroraWright/owocr/master/owocr_config.ini)

# Acknowledgments

This uses code from/references these projects:
- [Manga OCR](https://github.com/kha-white/manga-ocr)
- [ocrmac](https://github.com/straussmaximilian/ocrmac) for the Apple Vision framework API
- [NadeOCR](https://github.com/Natsume-197/NadeOCR) for the Google Vision API
- [ccylin2000_lipboard_monitor](https://github.com/vaimalaviya1233/ccylin2000_lipboard_monitor) for the Windows clipboard polling code

Thanks to viola for working on the Google Lens implementation!