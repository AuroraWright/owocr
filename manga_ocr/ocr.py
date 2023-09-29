import re
import os
import io
from pathlib import Path
import warnings
import configparser
import time
import sys
import platform

import jaconv
import torch
from PIL import Image
from loguru import logger
from transformers import ViTImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

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

class MangaOcr:
    def __init__(self, pretrained_model_name_or_path='kha-white/manga-ocr-base', force_cpu=False):
        logger.info(f'Loading OCR model from {pretrained_model_name_or_path}')
        self.processor = ViTImageProcessor.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(pretrained_model_name_or_path)

        if not force_cpu and torch.cuda.is_available():
            logger.info('Using CUDA')
            self.model.cuda()
        elif not force_cpu and torch.backends.mps.is_available():
            logger.info('Using MPS')
            warnings.filterwarnings("ignore", message=".*MPS: no support.*")
            self.model.to('mps')
        else:
            logger.info('Using CPU')

        logger.info('Manga OCR ready')

    def __call__(self, img_or_path):
        if isinstance(img_or_path, str) or isinstance(img_or_path, Path):
            img = Image.open(img_or_path)
        elif isinstance(img_or_path, Image.Image):
            img = img_or_path
        else:
            raise ValueError(f'img_or_path must be a path or PIL.Image, instead got: {img_or_path}')

        img = img.convert('L').convert('RGB')

        x = self._preprocess(img)
        x = self.model.generate(x[None].to(self.model.device), max_length=300)[0].cpu()
        x = self.tokenizer.decode(x, skip_special_tokens=True)
        x = post_process(x)
        return x

    def _preprocess(self, img):
        pixel_values = self.processor(img, return_tensors="pt").pixel_values
        return pixel_values.squeeze()

class GoogleVision:
    def __init__(self):
        if 'google.cloud' not in sys.modules:
            logger.warning('google-cloud-vision not available, Google Vision will not work!')
            self.available = False
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
                self.available = False

    def __call__(self, img_or_path):
        if not self.available:
            return "Engine not available!"

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
        img.save(image_bytes, format=img.format)
        return image_bytes.getvalue()

class AppleVision:
    def __init__(self):
        if sys.platform != "darwin":
            logger.warning('Apple Vision is not supported on non-macOS platforms!')
            self.available = False
        elif int(platform.mac_ver()[0].split('.')[0]) < 13:
            logger.warning('Apple Vision is not supported on macOS older than Ventura/13.0!')
            self.available = False
        else:
            if 'objc' not in sys.modules:
                logger.warning('pyobjc not available, Apple Vision will not work!')
                self.available = False
            else:
                self.available = True
                logger.info('Apple Vision ready')

    def __call__(self, img_or_path):
        if not self.available:
            return "Engine not available!"

        if isinstance(img_or_path, str) or isinstance(img_or_path, Path):
            img = Image.open(img_or_path)
        elif isinstance(img_or_path, Image.Image):
            img = img_or_path
        else:
            raise ValueError(f'img_or_path must be a path or PIL.Image, instead got: {img_or_path}')

        with objc.autorelease_pool():
            req = Vision.VNRecognizeTextRequest.alloc().init()

            req.setRecognitionLevel_(0)
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
        img.save(image_bytes, format=img.format)
        return image_bytes.getvalue()

class AzureComputerVision:
    def __init__(self):
        if 'azure.cognitiveservices.vision.computervision' not in sys.modules:
            logger.warning('azure-cognitiveservices-vision-computervision not available, Azure Computer Vision will not work!')
            self.available = False
        else:
            logger.info(f'Parsing Azure credentials')
            azure_credentials_file = os.path.join(os.path.expanduser('~'),'.config','azure_computer_vision.ini')
            try:
                azure_credentials = configparser.ConfigParser()
                azure_credentials.read(azure_credentials_file)
                self.client = ComputerVisionClient(azure_credentials['config']['endpoint'], CognitiveServicesCredentials(azure_credentials['config']['api_key']))
                self.available = True
                logger.info('Azure Computer Vision ready')
            except:
                logger.warning('Error parsing Azure credentials, Azure Computer Vision will not work!')
                self.available = False

    def __call__(self, img_or_path):
        if not self.available:
            return "Engine not available!"

        if isinstance(img_or_path, str) or isinstance(img_or_path, Path):
            img = Image.open(img_or_path)
        elif isinstance(img_or_path, Image.Image):
            img = img_or_path
        else:
            raise ValueError(f'img_or_path must be a path or PIL.Image, instead got: {img_or_path}')

        image_io = self._preprocess(img)
        read_response = self.client.read_in_stream(image_io, raw=True)

        read_operation_location = read_response.headers["Operation-Location"]
        operation_id = read_operation_location.split("/")[-1]

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
        img.save(image_io, format=img.format)
        image_io.seek(0)
        return image_io


def post_process(text):
    text = ''.join(text.split())
    text = text.replace('…', '...')
    text = re.sub('[・.]{2,}', lambda x: (x.end() - x.start()) * '.', text)
    text = jaconv.h2z(text, ascii=True, digit=True)

    return text
