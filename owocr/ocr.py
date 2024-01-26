import re
import os
import io
from pathlib import Path
import time
import sys
import platform
from math import sqrt

import jaconv
import numpy as np
import json
from PIL import Image
from loguru import logger

try:
    from manga_ocr import MangaOcr as MOCR
except ImportError:
    pass

try:
    import Vision
    import objc
except ImportError:
    pass

try:
    from google.cloud import vision
    from google.oauth2 import service_account
except ImportError:
    pass

try:
    from azure.cognitiveservices.vision.computervision import ComputerVisionClient
    from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
    from msrest.authentication import CognitiveServicesCredentials
except ImportError:
    pass

try:
    import easyocr
except ImportError:
    pass

try:
    from paddleocr import PaddleOCR as POCR
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


def post_process(text):
    text = ''.join(text.split())
    text = text.replace('…', '...')
    text = re.sub('[・.]{2,}', lambda x: (x.end() - x.start()) * '.', text)
    text = jaconv.h2z(text, ascii=True, digit=True)

    return text


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

        x = self.model(img)
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
        response = self.client.text_detection(image=image)
        texts = response.text_annotations
        x = post_process(texts[0].description)
        return x

    def _preprocess(self, img):
        image_bytes = io.BytesIO()
        img.save(image_bytes, format='png')
        return image_bytes.getvalue()

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
        files = {'encoded_image': ('owo' + str(timestamp) + '.png', self._preprocess(img), 'image/png')}
        try:
            res = requests.post(url, files=files, timeout=30)
        except requests.exceptions.Timeout:
            return 'Request timeout!'

        x = ''
        if res.status_code == 200:
            regex = re.compile(r">AF_initDataCallback\(({key: 'ds:1'.*?)\);</script>")
            match = regex.search(res.text)
            if match != None:
                lens_object = pyjson5.loads(match.group(1))
                if not 'errorHasStatus' in lens_object:
                    text = lens_object['data'][3][4][0]
                    if len(text) > 0:
                        lines = text[0]
                        for line in lines:
                            x += line + ' '
                        x = post_process(x)

        return x

    def _preprocess(self, img):
        w,h = img.size
        if w * h > 3000000:
            aspect_ratio = w/h
            new_w = int(sqrt(3000000 * aspect_ratio))
            new_h = int(new_w / aspect_ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        image_bytes = io.BytesIO()
        img.save(image_bytes, format='png')
        return image_bytes.getvalue()

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

            req.setRecognitionLevel_(0)
            req.setUsesLanguageCorrection_(True)
            req.setRecognitionLanguages_(['ja','en'])

            handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(
                self._preprocess(img), None
            )

            success = handler.performRequests_error_([req], None)
            res = ''
            if success[0]:
                for result in req.results():
                    res += result.text() + ' '
                req.dealloc()

            handler.dealloc()
            x = post_process(res)
            return x

    def _preprocess(self, img):
        image_bytes = io.BytesIO()
        img.save(image_bytes, format='png')
        return image_bytes.getvalue()

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
                return 'Request timeout!'

            res = json.loads(res.text)['text']

        x = post_process(res)
        return x

    def _preprocess(self, img):
        image_bytes = io.BytesIO()
        img.save(image_bytes, format='png')
        return image_bytes.getvalue()

class AzureComputerVision:
    name = 'azure'
    readable_name = 'Azure Computer Vision'
    key = 'v'
    available = False

    def __init__(self, config={}):
        if 'azure.cognitiveservices.vision.computervision' not in sys.modules:
            logger.warning('azure-cognitiveservices-vision-computervision not available, Azure Computer Vision will not work!')
        else:
            logger.info(f'Parsing Azure credentials')
            try:
                self.client = ComputerVisionClient(config['endpoint'], CognitiveServicesCredentials(config['api_key']))
                self.available = True
                logger.info('Azure Computer Vision ready')
            except:
                logger.warning('Error parsing Azure credentials, Azure Computer Vision will not work!')

    def __call__(self, img_or_path):
        if isinstance(img_or_path, str) or isinstance(img_or_path, Path):
            img = Image.open(img_or_path)
        elif isinstance(img_or_path, Image.Image):
            img = img_or_path
        else:
            raise ValueError(f'img_or_path must be a path or PIL.Image, instead got: {img_or_path}')

        image_io = self._preprocess(img)
        read_response = self.client.read_in_stream(image_io, raw=True)

        read_operation_location = read_response.headers['Operation-Location']
        operation_id = read_operation_location.split('/')[-1]

        while True:
            read_result = self.client.get_read_result(operation_id)
            if read_result.status.lower() not in ['notstarted', 'running']:
                break
            time.sleep(0.3)

        res = ''
        if read_result.status == OperationStatusCodes.succeeded:
            for text_result in read_result.analyze_result.read_results:
                for line in text_result.lines:
                    res += line.text + ' '

        x = post_process(res)
        return x

    def _preprocess(self, img):
        image_io = io.BytesIO()
        img.save(image_io, format='png')
        image_io.seek(0)
        return image_io

class EasyOCR:
    name = 'easyocr'
    readable_name = 'EasyOCR'
    key = 'e'
    available = False

    def __init__(self):
        if 'easyocr' not in sys.modules:
            logger.warning('easyocr not available, EasyOCR will not work!')
        else:
            logger.info('Loading EasyOCR model')
            self.model = easyocr.Reader(['ja','en'])
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
            res += text + ' '

        x = post_process(res)
        return x

    def _preprocess(self, img):
        image_bytes = io.BytesIO()
        img.save(image_bytes, format='png')
        return image_bytes.getvalue()

class PaddleOCR:
    name = 'paddleocr'
    readable_name = 'PaddleOCR'
    key = 'o'
    available = False

    def __init__(self):
        if 'paddleocr' not in sys.modules:
            logger.warning('paddleocr not available, PaddleOCR will not work!')
        else:
            logger.info('Loading PaddleOCR model')
            self.model = POCR(use_angle_cls=True, show_log=False, lang='japan')
            self.available = True
            logger.info('PaddleOCR ready')

    def __call__(self, img_or_path):
        if isinstance(img_or_path, str) or isinstance(img_or_path, Path):
            img = Image.open(img_or_path)
        elif isinstance(img_or_path, Image.Image):
            img = img_or_path
        else:
            raise ValueError(f'img_or_path must be a path or PIL.Image, instead got: {img_or_path}')

        res = ''
        read_results = self.model.ocr(self._preprocess(img), cls=True)
        for read_result in read_results:
            if read_result:
                for text in read_result:
                    res += text[1][0] + ' '

        x = post_process(res)
        return x

    def _preprocess(self, img):
        return np.array(img.convert('RGB'))
