# OwOCR

Command line client for several Japanese OCR providers derived from [Manga OCR](https://github.com/kha-white/manga-ocr).

# Installation

This has been tested with Python 3.11. Newer/older versions might work. For now it can be installed with `pip install https://github.com/AuroraWright/owocr/archive/master.zip`

# Supported providers

## Local providers
- [Manga OCR](https://github.com/kha-white/manga-ocr): refer to the readme for installation ("m" key)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR): refer to the readme for installation ("e" key)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR): refer to the [wiki](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_en/quickstart_en.md) for installation ("o" key)
- Apple Vision framework: this will work on macOS Ventura or later if pyobjc (`pip install pyobjc`) is installed. In my experience, the best of the local providers for horizontal text ("a" key)
- WinRT OCR: this will work on Windows 10 or later if winocr (`pip install winocr`) is installed. It can also be used by installing winocr on a Windows virtual machine and running the server (`winocr_serve`), installing requests (`pip install requests`) and specifying the IP address of the Windows VM/machine in the config file (see below) ("w" key)

## Cloud providers
- Google Vision: you need a service account .json file named google_vision.json in `user directory/.config/` and installing google-cloud-vision (`pip install google-cloud-vision`)
- Azure Computer Vision: you need to specify an api key and an endpoint in the config file (see below) and to install azure-cognitiveservices-vision-computervision (`pip install azure-cognitiveservices-vision-computervision`)

# Usage

It mostly functions like Manga OCR: https://github.com/kha-white/manga-ocr?tab=readme-ov-file#running-in-the-background
However:
- you can pause/unpause the clipboard image processing by pressing "p" or terminate the script with "t" or "q"
- you can switch OCR provider with its corresponding keyboard key (refer to the list above). You can also start the script paused with the -p option or with a specific provider with the -e option (refer to `owocr -h` for the list)
- holding ctrl or cmd at any time will pause the clipboard image processing temporarily
- for systems where text can be copied to the clipboard at the same time as images, if `*ocr_ignore*` is copied with an image, the image will be ignored
- a config file (located in `user directory/.config/owocr_config.ini`) can be used to limit providers (to reduce clutter/memory usage) as well as specifying provider settings such as api keys etc (a sample config file is provided)

# Acknowledgments

This uses code from/references these projects:
- [Manga OCR](https://github.com/kha-white/manga-ocr)
- [ocrmac](https://github.com/straussmaximilian/ocrmac) for the Apple Vision framework API
- [NadeOCR](https://github.com/Natsume-197/NadeOCR) for the Google Vision API
