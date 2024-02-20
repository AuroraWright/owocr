import re
import os
import io
from pathlib import Path
import time
import sys
import platform
import logging
from math import sqrt

import jaconv
import numpy as np
from PIL import Image
from loguru import logger

try:
    from manga_ocr import MangaOcr as MOCR
except ImportError:
    pass

try:
    import Vision
    import objc
    from AppKit import NSData, NSImage, NSBundle
    from CoreFoundation import CFRunLoopRun, CFRunLoopStop, CFRunLoopGetCurrent
except ImportError:
    pass

try:
    from google.cloud import vision
    from google.oauth2 import service_account
    from google.api_core.exceptions import ServiceUnavailable
except ImportError:
    pass

try:
    from azure.ai.vision.imageanalysis import ImageAnalysisClient
    from azure.ai.vision.imageanalysis.models import VisualFeatures
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import ServiceRequestError
except ImportError:
    pass

try:
    import easyocr
except ImportError:
    pass

try:
    from rapidocr_onnxruntime import RapidOCR as ROCR
    import urllib.request
except ImportError:
    pass

try:
    import requests
except ImportError:
    pass

try:
    import winocr
except ImportError:
    pass

try:
    import pyjson5
except ImportError:
    pass

try:
    import fpng_py
    optimized_png_encode = True
except:
    optimized_png_encode = False


def empty_post_process(text):
    return text


def post_process(text):
    text = ' '.join([''.join(i.split()) for i in text.splitlines()])
    text = text.replace('…', '...')
    text = re.sub('[・.]{2,}', lambda x: (x.end() - x.start()) * '.', text)
    text = jaconv.h2z(text, ascii=True, digit=True)
    return text


def pil_image_to_bytes(img, img_format='png', png_compression=6):
    if img_format == 'png' and optimized_png_encode:
        raw_data = img.convert('RGBA').tobytes()
        image_bytes = fpng_py.fpng_encode_image_to_memory(raw_data, img.width, img.height)
    else:
        image_bytes = io.BytesIO()
        img.save(image_bytes, format=img_format, compress_level=png_compression)
        image_bytes = image_bytes.getvalue()
    return image_bytes


def pil_image_to_numpy_array(img):
    return np.array(img.convert('RGBA'))


class MangaOcr:
    name = 'mangaocr'
    readable_name = 'Manga OCR'
    key = 'm'
    available = False

    def __init__(self, config={'pretrained_model_name_or_path':'kha-white/manga-ocr-base','force_cpu': False}):
        if 'manga_ocr' not in sys.modules:
            logger.warning('manga-ocr not available, Manga OCR will not work!')
        else:
            logger.disable('manga_ocr')
            from manga_ocr import ocr
            ocr.post_process = empty_post_process
            logger.info(f'Loading Manga OCR model')
            self.model = MOCR(config['pretrained_model_name_or_path'], config['force_cpu'])
            self.available = True
            logger.info('Manga OCR ready')

    def __call__(self, img_or_path):
        if isinstance(img_or_path, str) or isinstance(img_or_path, Path):
            img = Image.open(img_or_path)
        elif isinstance(img_or_path, Image.Image):
            img = img_or_path
        else:
            raise ValueError(f'img_or_path must be a path or PIL.Image, instead got: {img_or_path}')

        x = (True, self.model(img))
        return x

class GoogleVision:
    name = 'gvision'
    readable_name = 'Google Vision'
    key = 'g'
    available = False

    def __init__(self):
        if 'google.cloud' not in sys.modules:
            logger.warning('google-cloud-vision not available, Google Vision will not work!')
        else:
            logger.info(f'Parsing Google credentials')
            google_credentials_file = os.path.join(os.path.expanduser('~'),'.config','google_vision.json')
            try:
                google_credentials = service_account.Credentials.from_service_account_file(google_credentials_file)
                self.client = vision.ImageAnnotatorClient(credentials=google_credentials)
                self.available = True
                logger.info('Google Vision ready')
            except:
                logger.warning('Error parsing Google credentials, Google Vision will not work!')

    def __call__(self, img_or_path):
        if isinstance(img_or_path, str) or isinstance(img_or_path, Path):
            img = Image.open(img_or_path)
        elif isinstance(img_or_path, Image.Image):
            img = img_or_path
        else:
            raise ValueError(f'img_or_path must be a path or PIL.Image, instead got: {img_or_path}')

        image_bytes = self._preprocess(img)
        image = vision.Image(content=image_bytes)
        try:
            response = self.client.text_detection(image=image)
        except ServiceUnavailable:
            return (False, 'Connection error!')
        except:
            return (False, 'Unknown error!')
        texts = response.text_annotations
        res = texts[0].description if len(texts) > 0 else ''
        x = (True, res)
        return x

    def _preprocess(self, img):
        return pil_image_to_bytes(img)

class GoogleLens:
    name = 'glens'
    readable_name = 'Google Lens'
    key = 'l'
    available = False

    def __init__(self):
        if 'pyjson5' not in sys.modules:
            logger.warning('pyjson5 not available, Google Lens will not work!')
        elif 'requests' not in sys.modules:
            logger.warning('requests not available, Google Lens will not work!')
        else:
            self.regex = re.compile(r">AF_initDataCallback\(({key: 'ds:1'.*?)\);</script>")
            self.available = True
            logger.info('Google Lens ready')

    def __call__(self, img_or_path):
        if isinstance(img_or_path, str) or isinstance(img_or_path, Path):
            img = Image.open(img_or_path)
        elif isinstance(img_or_path, Image.Image):
            img = img_or_path
        else:
            raise ValueError(f'img_or_path must be a path or PIL.Image, instead got: {img_or_path}')

        timestamp = int(time.time() * 1000)
        url = f'https://lens.google.com/v3/upload?stcs={timestamp}'
        headers = {'User-Agent': 'Mozilla/5.0 (Linux; Android 13; RMX3771) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.6167.144 Mobile Safari/537.36'}
        cookies = {'SOCS': 'CAESEwgDEgk0ODE3Nzk3MjQaAmVuIAEaBgiA_LyaBg'}
        files = {'encoded_image': ('owo' + str(timestamp) + '.png', self._preprocess(img), 'image/png')}
        try:
            res = requests.post(url, files=files, headers=headers, cookies=cookies, timeout=20)
        except requests.exceptions.Timeout:
            return (False, 'Request timeout!')
        except requests.exceptions.ConnectionError:
            return (False, 'Connection error!')

        if res.status_code != 200:
            return (False, 'Unknown error!')

        match = self.regex.search(res.text)
        if match == None:
            return (False, 'Regex error!')

        lens_object = pyjson5.loads(match.group(1))
        if 'errorHasStatus' in lens_object:
            return (False, 'Unknown Lens error!')

        res = ''
        text = lens_object['data'][3][4][0]
        if len(text) > 0:
            lines = text[0]
            for line in lines:
                res += line + '\n'

        x = (True, res)
        return x

    def _preprocess(self, img):
        if img.width * img.height > 3000000:
            aspect_ratio = img.width / img.height
            new_w = int(sqrt(3000000 * aspect_ratio))
            new_h = int(new_w / aspect_ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        return pil_image_to_bytes(img)

class AppleVision:
    name = 'avision'
    readable_name = 'Apple Vision'
    key = 'a'
    available = False

    def __init__(self):
        if sys.platform != 'darwin':
            logger.warning('Apple Vision is not supported on non-macOS platforms!')
        elif int(platform.mac_ver()[0].split('.')[0]) < 13:
            logger.warning('Apple Vision is not supported on macOS older than Ventura/13.0!')
        else:
            self.available = True
            logger.info('Apple Vision ready')

    def __call__(self, img_or_path):
        if isinstance(img_or_path, str) or isinstance(img_or_path, Path):
            img = Image.open(img_or_path)
        elif isinstance(img_or_path, Image.Image):
            img = img_or_path
        else:
            raise ValueError(f'img_or_path must be a path or PIL.Image, instead got: {img_or_path}')

        with objc.autorelease_pool():
            req = Vision.VNRecognizeTextRequest.alloc().init()

            req.setRevision_(Vision.VNRecognizeTextRequestRevision3)
            req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
            req.setUsesLanguageCorrection_(True)
            req.setRecognitionLanguages_(['ja','en'])

            handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(
                self._preprocess(img), None
            )

            success = handler.performRequests_error_([req], None)
            res = ''
            if success[0]:
                for result in req.results():
                    res += result.text() + '\n'
                x = (True, res)
            else:
                x = (False, 'Unknown error!')

            return x

    def _preprocess(self, img):
        return pil_image_to_bytes(img, 'tiff')


class AppleLiveText:
    name = 'alivetext'
    readable_name = 'Apple Live Text'
    key = 'd'
    available = False

    def __init__(self):
        if sys.platform != 'darwin':
            logger.warning('Apple Live Text is not supported on non-macOS platforms!')
        elif int(platform.mac_ver()[0].split('.')[0]) < 13:
            logger.warning('Apple Live Text is not supported on macOS older than Ventura/13.0!')
        else:
            app_info = NSBundle.mainBundle().infoDictionary()
            app_info['LSBackgroundOnly'] = '1'
            objc.loadBundle('VisionKit', globals(), '/System/Library/Frameworks/VisionKit.framework')
            objc.registerMetaDataForSelector(
                b'VKCImageAnalyzer',
                b'processRequest:progressHandler:completionHandler:',
                {
                    'arguments': {
                        3: {
                            'callable': {
                                'retval': {'type': b'v'},
                                'arguments': {
                                    0: {'type': b'^v'},
                                    1: {'type': b'd'},
                                }
                            }
                        },
                        4: {
                            'callable': {
                                'retval': {'type': b'v'},
                                'arguments': {
                                    0: {'type': b'^v'},
                                    1: {'type': b'@'},
                                    2: {'type': b'@'},
                                }
                            }
                        }
                    }
                }
            )
            self.analyzer = VKCImageAnalyzer.alloc().init()
            self.available = True
            logger.info('Apple Live Text ready')

    def __call__(self, img_or_path):
        if isinstance(img_or_path, str) or isinstance(img_or_path, Path):
            img = Image.open(img_or_path)
        elif isinstance(img_or_path, Image.Image):
            img = img_or_path
        else:
            raise ValueError(f'img_or_path must be a path or PIL.Image, instead got: {img_or_path}')

        req = VKCImageAnalyzerRequest.alloc().initWithImage_requestType_(self._preprocess(img), 1) #VKAnalysisTypeText
        req.setLocales_(['ja','en'])
        self.result = None
        self.analyzer.processRequest_progressHandler_completionHandler_(req, lambda progress: None, self._process)

        CFRunLoopRun()

        if self.result == None:
            return (False, 'Unknown error!')
        return (True, self.result)

    def _process(self, analysis, error):
        res = ''
        lines = analysis.allLines()
        if lines:
            for line in lines:
                res += line.string() + '\n'
        self.result = res
        CFRunLoopStop(CFRunLoopGetCurrent())

    def _preprocess(self, img):
        image_bytes = pil_image_to_bytes(img, 'tiff')
        ns_data = NSData.dataWithBytes_length_(image_bytes, len(image_bytes))
        ns_image = NSImage.alloc().initWithData_(ns_data)
        return ns_image


class WinRTOCR:
    name = 'winrtocr'
    readable_name = 'WinRT OCR'
    key = 'w'
    available = False

    def __init__(self, config={}):
        if sys.platform == 'win32':
            if int(platform.release()) < 10:
                logger.warning('WinRT OCR is not supported on Windows older than 10!')
            elif 'winocr' not in sys.modules:
                logger.warning('winocr not available, WinRT OCR will not work!')
            else:
                self.available = True
                logger.info('WinRT OCR ready')
        else:
            if 'requests' not in sys.modules:
                logger.warning('requests not available, WinRT OCR will not work!')
            else:
                try:
                    self.url = config['url']
                    self.available = True
                    logger.info('WinRT OCR ready')
                except:
                    logger.warning('Error reading URL from config, WinRT OCR will not work!')

    def __call__(self, img_or_path):
        if isinstance(img_or_path, str) or isinstance(img_or_path, Path):
            img = Image.open(img_or_path)
        elif isinstance(img_or_path, Image.Image):
            img = img_or_path
        else:
            raise ValueError(f'img_or_path must be a path or PIL.Image, instead got: {img_or_path}')

        if sys.platform == 'win32':
            res = winocr.recognize_pil_sync(img, lang='ja')['text']
        else:
            params = {'lang': 'ja'}
            try:
                res = requests.post(self.url, params=params, data=self._preprocess(img), timeout=3)
            except requests.exceptions.Timeout:
                return (False, 'Request timeout!')
            except requests.exceptions.ConnectionError:
                return (False, 'Connection error!')

            if res.status_code != 200:
                return (False, 'Unknown error!')

            res = res.json()['text']

        x = (True, res)
        return x

    def _preprocess(self, img):
        return pil_image_to_bytes(img, png_compression=1)

class AzureImageAnalysis:
    name = 'azure'
    readable_name = 'Azure Image Analysis'
    key = 'v'
    available = False

    def __init__(self, config={}):
        if 'azure.ai.vision.imageanalysis' not in sys.modules:
            logger.warning('azure-ai-vision-imageanalysis not available, Azure Image Analysis will not work!')
        else:
            logger.info(f'Parsing Azure credentials')
            try:
                self.client = ImageAnalysisClient(config['endpoint'], AzureKeyCredential(config['api_key']))
                self.available = True
                logger.info('Azure Image Analysis ready')
            except:
                logger.warning('Error parsing Azure credentials, Azure Image Analysis will not work!')

    def __call__(self, img_or_path):
        if isinstance(img_or_path, str) or isinstance(img_or_path, Path):
            img = Image.open(img_or_path)
        elif isinstance(img_or_path, Image.Image):
            img = img_or_path
        else:
            raise ValueError(f'img_or_path must be a path or PIL.Image, instead got: {img_or_path}')

        try:
            read_result = self.client.analyze(image_data=self._preprocess(img), visual_features=[VisualFeatures.READ])
        except ServiceRequestError:
            return (False, 'Connection error!')
        except:
            return (False, 'Unknown error!')

        res = ''
        if read_result.read:
            for block in read_result.read.blocks:
                for line in block.lines:
                    res += line.text + '\n'
        else:
            return (False, 'Unknown error!')

        x = (True, res)
        return x

    def _preprocess(self, img):
        if any(x < 50 for x in img.size):
            resize_factor = max(50 / img.width, 50 / img.height)
            new_w = int(img.width * resize_factor)
            new_h = int(img.height * resize_factor)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        return pil_image_to_bytes(img)

class EasyOCR:
    name = 'easyocr'
    readable_name = 'EasyOCR'
    key = 'e'
    available = False

    def __init__(self, config={'gpu': True}):
        if 'easyocr' not in sys.modules:
            logger.warning('easyocr not available, EasyOCR will not work!')
        else:
            logger.info('Loading EasyOCR model')
            logging.getLogger('easyocr.easyocr').disabled = True
            self.model = easyocr.Reader(['ja','en'], gpu=config['gpu'])
            self.available = True
            logger.info('EasyOCR ready')

    def __call__(self, img_or_path):
        if isinstance(img_or_path, str) or isinstance(img_or_path, Path):
            img = Image.open(img_or_path)
        elif isinstance(img_or_path, Image.Image):
            img = img_or_path
        else:
            raise ValueError(f'img_or_path must be a path or PIL.Image, instead got: {img_or_path}')

        res = ''
        read_result = self.model.readtext(self._preprocess(img), detail=0)
        for text in read_result:
            res += text + '\n'

        x = (True, res)
        return x

    def _preprocess(self, img):
        return pil_image_to_numpy_array(img)

class RapidOCR:
    name = 'rapidocr'
    readable_name = 'RapidOCR'
    key = 'r'
    available = False

    def __init__(self):
        if 'rapidocr_onnxruntime' not in sys.modules:
            logger.warning('rapidocr_onnxruntime not available, RapidOCR will not work!')
        else:
            rapidocr_model_file = os.path.join(os.path.expanduser('~'),'.cache','rapidocr_japan_PP-OCRv4_rec_infer.onnx')
            if not os.path.isfile(rapidocr_model_file):
                logger.info('Downloading RapidOCR model ' + rapidocr_model_file)
                try:
                    cache_folder = os.path.join(os.path.expanduser('~'),'.cache')
                    if not os.path.isdir(cache_folder):
                        os.makedirs(cache_folder)
                    urllib.request.urlretrieve('https://github.com/AuroraWright/owocr/raw/master/rapidocr_japan_PP-OCRv4_rec_infer.onnx', rapidocr_model_file)
                except:
                    logger.warning('Download failed. RapidOCR will not work!')
                    return

            logger.info('Loading RapidOCR model')
            self.model = ROCR(rec_model_path=rapidocr_model_file)
            logging.getLogger().disabled = True
            self.available = True
            logger.info('RapidOCR ready')

    def __call__(self, img_or_path):
        if isinstance(img_or_path, str) or isinstance(img_or_path, Path):
            img = Image.open(img_or_path)
        elif isinstance(img_or_path, Image.Image):
            img = img_or_path
        else:
            raise ValueError(f'img_or_path must be a path or PIL.Image, instead got: {img_or_path}')

        res = ''
        read_results, elapsed = self.model(self._preprocess(img))
        if read_results:
            for read_result in read_results:
                res += read_result[1] + '\n'

        x = (True, res)
        return x

    def _preprocess(self, img):
        return pil_image_to_numpy_array(img)
