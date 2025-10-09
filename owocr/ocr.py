import re
import os
import io
from pathlib import Path
import sys
import platform
import logging
from math import sqrt, sin, cos, atan2
import json
import base64
from urllib.parse import urlparse, parse_qs
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import jaconv
import numpy as np
from PIL import Image
from loguru import logger
import requests

try:
    from manga_ocr import MangaOcr as MOCR
except ImportError:
    pass

try:
    import Vision
    import objc
    from AppKit import NSData, NSImage, NSBundle
    from CoreFoundation import CFRunLoopRunInMode, kCFRunLoopDefaultMode, CFRunLoopStop, CFRunLoopGetCurrent
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
    from rapidocr import RapidOCR as ROCR
    from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion
except ImportError:
    pass

try:
    import winocr
except ImportError:
    pass

try:
    import oneocr
except ImportError:
    pass

try:
    import pyjson5
except ImportError:
    pass

try:
    import betterproto
    from .lens_betterproto import *
    import random
except ImportError:
    pass

try:
    import fpng_py
    optimized_png_encode = True
except:
    optimized_png_encode = False


@dataclass
class BoundingBox:
    """
    Represents the normalized coordinates of a detected element.
    All values are floats between 0.0 and 1.0.
    """
    center_x: float
    center_y: float
    width: float
    height: float
    rotation_z: Optional[float] = None  # Optional rotation in radians

@dataclass
class Word:
    """Represents a single recognized word and its properties."""
    text: str
    bounding_box: BoundingBox
    separator: Optional[str] = None  # The character(s) that follow the word, e.g., a space

@dataclass
class Line:
    """Represents a single line of text, composed of words."""
    bounding_box: BoundingBox
    words: List[Word] = field(default_factory=list)
    text: Optional[str] = None

@dataclass
class Paragraph:
    """Represents a block of text, composed of lines."""
    bounding_box: BoundingBox
    lines: List[Line] = field(default_factory=list)
    writing_direction: Optional[str] = None # Optional: e.g., "LEFT_TO_RIGHT"

@dataclass
class ImageProperties:
    """Stores the original dimensions of the processed image."""
    width: int
    height: int

@dataclass
class OcrResult:
    """The root object for a complete OCR analysis of an image."""
    image_properties: ImageProperties
    paragraphs: List[Paragraph] = field(default_factory=list)


def empty_post_process(text):
    return text

def input_to_pil_image(img):
    is_path = False
    if isinstance(img, Image.Image):
        pil_image = img
    elif isinstance(img, (bytes, bytearray)):
        pil_image = Image.open(io.BytesIO(img))
    elif isinstance(img, Path):
        is_path = True
        try:
            pil_image = Image.open(img)
            pil_image.load()
        except (UnidentifiedImageError, OSError) as e:
            return None
    else:
        raise ValueError(f'img must be a path, PIL.Image or bytes object, instead got: {img}')
    return pil_image, is_path

def pil_image_to_bytes(img, img_format='png', png_compression=6, jpeg_quality=80, optimize=False):
    if img_format == 'png' and optimized_png_encode and not optimize:
        raw_data = img.convert('RGBA').tobytes()
        image_bytes = fpng_py.fpng_encode_image_to_memory(raw_data, img.width, img.height)
    else:
        image_bytes = io.BytesIO()
        if img_format == 'jpeg':
            img = img.convert('RGB')
        img.save(image_bytes, format=img_format, compress_level=png_compression, quality=jpeg_quality, optimize=optimize, subsampling=0)
        image_bytes = image_bytes.getvalue()
    return image_bytes

def pil_image_to_numpy_array(img):
    return np.array(img.convert('RGBA'))

def limit_image_size(img, max_size):
    img_bytes = pil_image_to_bytes(img)
    if len(img_bytes) <= max_size:
        return img_bytes, 'png'

    scaling_factor = 0.60 if any(x > 2000 for x in img.size) else 0.75
    new_w = int(img.width * scaling_factor)
    new_h = int(img.height * scaling_factor)
    resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized_img_bytes = pil_image_to_bytes(resized_img)
    if len(resized_img_bytes) <= max_size:
        return resized_img_bytes, 'png'

    for _ in range(2):
        jpeg_quality = 80
        while jpeg_quality >= 60:
            img_bytes = pil_image_to_bytes(img, 'jpeg', jpeg_quality=jpeg_quality, optimize=True)
            if len(img_bytes) <= max_size:
                return img_bytes, 'jpeg'
            jpeg_quality -= 5
        img = resized_img

    return False, ''


class MangaOcr:
    name = 'mangaocr'
    readable_name = 'Manga OCR'
    key = 'm'
    available = False
    local = True
    manual_language = False
    coordinate_support = False

    def __init__(self, config={'pretrained_model_name_or_path':'kha-white/manga-ocr-base','force_cpu': False}):
        if 'manga_ocr' not in sys.modules:
            logger.warning('manga-ocr not available, Manga OCR will not work!')
        else:
            logger.disable('manga_ocr')
            logging.getLogger('transformers').setLevel(logging.ERROR) # silence transformers >=4.46 warnings
            from manga_ocr import ocr
            ocr.post_process = empty_post_process
            logger.info(f'Loading Manga OCR model')
            self.model = MOCR(config['pretrained_model_name_or_path'], config['force_cpu'])
            self.available = True
            logger.info('Manga OCR ready')

    def __call__(self, img):
        img, is_path = input_to_pil_image(img)
        if not img:
            return (False, 'Invalid image provided')

        x = (True, self.model(img))

        if is_path:
            img.close()
        return x

class GoogleVision:
    name = 'gvision'
    readable_name = 'Google Vision'
    key = 'g'
    available = False
    local = False
    manual_language = False
    coordinate_support = False

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

    def __call__(self, img):
        img, is_path = input_to_pil_image(img)
        if not img:
            return (False, 'Invalid image provided')

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

        if is_path:
            img.close()
        return x

    def _preprocess(self, img):
        return pil_image_to_bytes(img)

class GoogleLens:
    name = 'glens'
    readable_name = 'Google Lens'
    key = 'l'
    available = False
    local = False
    manual_language = False
    coordinate_support = True

    def __init__(self):
        if 'betterproto' not in sys.modules:
            logger.warning('betterproto not available, Google Lens will not work!')
        else:
            self.available = True
            logger.info('Google Lens ready')

    def _to_generic_result(self, response, img_width, img_height):
        paragraphs = []
        if 'objects_response' in response and 'text' in response['objects_response']:
            text_data = response['objects_response']['text']
            if 'text_layout' in text_data:
                for p in text_data['text_layout'].get('paragraphs', []):
                    lines = []
                    for l in p.get('lines', []):
                        words = []
                        for w in l.get('words', []):
                            w_bbox = w.get('geometry', {}).get('bounding_box', {})
                            word = Word(
                                text=w.get('plain_text', ''),
                                separator=w.get('text_separator'),
                                bounding_box=BoundingBox(
                                    center_x=w_bbox.get('center_x'),
                                    center_y=w_bbox.get('center_y'),
                                    width=w_bbox.get('width'),
                                    height=w_bbox.get('height'),
                                    rotation_z=w_bbox.get('rotation_z')
                                )
                            )
                            words.append(word)
                        
                        l_bbox = l.get('geometry', {}).get('bounding_box', {})
                        line = Line(
                            bounding_box=BoundingBox(
                                center_x=l_bbox.get('center_x'),
                                center_y=l_bbox.get('center_y'),
                                width=l_bbox.get('width'),
                                height=l_bbox.get('height'),
                                rotation_z=l_bbox.get('rotation_z')
                            ),
                            words=words
                        )
                        lines.append(line)

                    p_bbox = p.get('geometry', {}).get('bounding_box', {})
                    paragraph = Paragraph(
                        bounding_box=BoundingBox(
                            center_x=p_bbox.get('center_x'),
                            center_y=p_bbox.get('center_y'),
                            width=p_bbox.get('width'),
                            height=p_bbox.get('height'),
                            rotation_z=p_bbox.get('rotation_z')
                        ),
                        lines=lines,
                        writing_direction=p.get('writing_direction')
                    )
                    paragraphs.append(paragraph)

        return OcrResult(
            image_properties=ImageProperties(width=img_width, height=img_height),
            paragraphs=paragraphs
        )

    def __call__(self, img):
        img, is_path = input_to_pil_image(img)
        if not img:
            return (False, 'Invalid image provided')

        request = LensOverlayServerRequest()

        request.objects_request.request_context.request_id.uuid = random.randint(0, 2**64 - 1)
        request.objects_request.request_context.request_id.sequence_id = 0
        request.objects_request.request_context.request_id.image_sequence_id = 0
        request.objects_request.request_context.request_id.analytics_id = random.randbytes(16)
        request.objects_request.request_context.request_id.routing_info = LensOverlayRoutingInfo()

        request.objects_request.request_context.client_context.platform = Platform.WEB
        request.objects_request.request_context.client_context.surface = Surface.CHROMIUM

        request.objects_request.request_context.client_context.locale_context.language = 'ja'
        request.objects_request.request_context.client_context.locale_context.region = 'Asia/Tokyo'
        request.objects_request.request_context.client_context.locale_context.time_zone = '' # not set by chromium

        request.objects_request.request_context.client_context.app_id = '' # not set by chromium

        filter = AppliedFilter()
        filter.filter_type = LensOverlayFilterType.AUTO_FILTER
        request.objects_request.request_context.client_context.client_filters.filter.append(filter)

        image_data = self._preprocess(img)
        request.objects_request.image_data.payload.image_bytes = image_data[0]
        request.objects_request.image_data.image_metadata.width = image_data[1]
        request.objects_request.image_data.image_metadata.height = image_data[2]

        payload = request.SerializeToString()

        headers = {
            'Host': 'lensfrontend-pa.googleapis.com',
            'Connection': 'keep-alive',
            'Content-Type': 'application/x-protobuf',
            'X-Goog-Api-Key': 'AIzaSyDr2UxVnv_U85AbhhY8XSHSIavUW0DC-sY',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-Mode': 'no-cors',
            'Sec-Fetch-Dest': 'empty',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'Accept-Language': 'ja-JP;q=0.6,ja;q=0.5'
        }

        try:
            res = requests.post('https://lensfrontend-pa.googleapis.com/v1/crupload', data=payload, headers=headers, timeout=20)
        except requests.exceptions.Timeout:
            return (False, 'Request timeout!')
        except requests.exceptions.ConnectionError:
            return (False, 'Connection error!')

        if res.status_code != 200:
            return (False, 'Unknown error!')

        response_proto = LensOverlayServerResponse().FromString(res.content)
        response_dict = response_proto.to_dict(betterproto.Casing.SNAKE)

        ocr_result = self._to_generic_result(response_dict, img.width, img.height)
        x = (True, ocr_result)

        if is_path:
            img.close()
        return x

    def _preprocess(self, img):
        if img.width * img.height > 3000000:
            aspect_ratio = img.width / img.height
            new_w = int(sqrt(3000000 * aspect_ratio))
            new_h = int(new_w / aspect_ratio)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        return (pil_image_to_bytes(img), img.width, img.height)

class GoogleLensWeb:
    name = 'glensweb'
    readable_name = 'Google Lens (web)'
    key = 'k'
    available = False
    local = False
    manual_language = False
    coordinate_support = False

    def __init__(self):
        if 'pyjson5' not in sys.modules:
            logger.warning('pyjson5 not available, Google Lens (web) will not work!')
        else:
            self.requests_session = requests.Session()
            self.available = True
            logger.info('Google Lens (web) ready')

    def __call__(self, img):
        img, is_path = input_to_pil_image(img)
        if not img:
            return (False, 'Invalid image provided')

        url = 'https://lens.google.com/v3/upload'
        files = {'encoded_image': ('image.png', self._preprocess(img), 'image/png')}
        headers = {
            'Host': 'lens.google.com',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:136.0) Gecko/20100101 Firefox/136.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ja-JP;q=0.6,ja;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'Referer': 'https://www.google.com/',
            'Origin': 'https://www.google.com',
            'Alt-Used': 'lens.google.com',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'same-site',
            'Priority': 'u=0, i',
            'TE': 'trailers'
        }
        cookies = {'SOCS': 'CAESEwgDEgk0ODE3Nzk3MjQaAmVuIAEaBgiA_LyaBg'}

        try:
            res = self.requests_session.post(url, files=files, headers=headers, cookies=cookies, timeout=20, allow_redirects=False)
        except requests.exceptions.Timeout:
            return (False, 'Request timeout!')
        except requests.exceptions.ConnectionError:
            return (False, 'Connection error!')

        if res.status_code != 303:
            return (False, 'Unknown error!')

        redirect_url = res.headers.get('Location')
        if not redirect_url:
            return (False, 'Error getting redirect URL!')

        parsed_url = urlparse(redirect_url)
        query_params = parse_qs(parsed_url.query)

        if ('vsrid' not in query_params) or ('gsessionid' not in query_params):
            return (False, 'Unknown error!')

        try:
            res = self.requests_session.get(f"https://lens.google.com/qfmetadata?vsrid={query_params['vsrid'][0]}&gsessionid={query_params['gsessionid'][0]}", timeout=20)
        except requests.exceptions.Timeout:
            return (False, 'Request timeout!')
        except requests.exceptions.ConnectionError:
            return (False, 'Connection error!')

        if (len(res.text.splitlines()) != 3):
            return (False, 'Unknown error!')

        lens_object = pyjson5.loads(res.text.splitlines()[2])

        res = []
        text = lens_object[0][2][0][0]
        for paragraph in text:
            for line in paragraph[1]:
                for word in line[0]:
                    res.append(word[1] + word[2])

        x = (True, res)

        if is_path:
            img.close()
        return x

    def _preprocess(self, img):
        if img.width * img.height > 3000000:
            aspect_ratio = img.width / img.height
            new_w = int(sqrt(3000000 * aspect_ratio))
            new_h = int(new_w / aspect_ratio)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        return pil_image_to_bytes(img)

class Bing:
    name = 'bing'
    readable_name = 'Bing'
    key = 'b'
    available = False
    local = False
    manual_language = False
    coordinate_support = True

    def __init__(self):
        self.requests_session = requests.Session()
        self.available = True
        logger.info('Bing ready')

    def _quad_to_center_bbox(self, quad):
        center_x = (quad['topLeft']['x'] + quad['topRight']['x'] + quad['bottomRight']['x'] + quad['bottomLeft']['x']) / 4
        center_y = (quad['topLeft']['y'] + quad['topRight']['y'] + quad['bottomRight']['y'] + quad['bottomLeft']['y']) / 4
        
        width1 = sqrt((quad['topRight']['x'] - quad['topLeft']['x'])**2 + (quad['topRight']['y'] - quad['topLeft']['y'])**2)
        width2 = sqrt((quad['bottomRight']['x'] - quad['bottomLeft']['x'])**2 + (quad['bottomRight']['y'] - quad['bottomLeft']['y'])**2)
        avg_width = (width1 + width2) / 2

        height1 = sqrt((quad['bottomLeft']['x'] - quad['topLeft']['x'])**2 + (quad['bottomLeft']['y'] - quad['topLeft']['y'])**2)
        height2 = sqrt((quad['bottomRight']['x'] - quad['topRight']['x'])**2 + (quad['bottomRight']['y'] - quad['topRight']['y'])**2)
        avg_height = (height1 + height2) / 2
        
        return BoundingBox(center_x=center_x, center_y=center_y, width=avg_width, height=avg_height)

    def _to_generic_result(self, response, img_width, img_height):
        paragraphs = []
        text_tag = None
        for tag in response.get('tags', []):
            if tag.get('displayName') == '##TextRecognition':
                text_tag = tag
                break
        
        if text_tag:
            text_action = None
            for action in text_tag.get('actions', []):
                if action.get('_type') == 'ImageKnowledge/TextRecognitionAction':
                    text_action = action
                    break
            
            if text_action:
                for p in text_action.get('data', {}).get('regions', []):
                    lines = []
                    for l in p.get('lines', []):
                        words = []
                        for w in l.get('words', []):
                            word = Word(
                                text=w.get('text', ''),
                                bounding_box=self._quad_to_center_bbox(w['boundingBox'])
                            )
                            words.append(word)

                        line = Line(
                            text=l.get('text', ''),
                            bounding_box=self._quad_to_center_bbox(l['boundingBox']),
                            words=words
                        )
                        lines.append(line)
                    
                    # Bing doesn't provide paragraph-level separators, so we add a newline
                    if lines and lines[-1].words:
                        lines[-1].words[-1].separator = '\n'

                    paragraph = Paragraph(
                        bounding_box=self._quad_to_center_bbox(p['boundingBox']),
                        lines=lines
                    )
                    paragraphs.append(paragraph)

        return OcrResult(
            image_properties=ImageProperties(width=img_width, height=img_height),
            paragraphs=paragraphs
        )

    def __call__(self, img):
        img, is_path = input_to_pil_image(img)
        if not img:
            return (False, 'Invalid image provided')

        img_bytes = self._preprocess(img)
        if not img_bytes:
            return (False, 'Image is too big!')

        upload_url = 'https://www.bing.com/images/search?view=detailv2&iss=sbiupload'
        upload_headers = {
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'accept-language': 'ja-JP;q=0.6,ja;q=0.5',
            'cache-control': 'max-age=0',
            'origin': 'https://www.bing.com',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:136.0) Gecko/20100101 Firefox/136.0',
        }
        files = {
            'imgurl': (None, ''),
            'cbir': (None, 'sbi'),
            'imageBin': (None, img_bytes)
        }

        for _ in range(2):
            api_host = urlparse(upload_url).netloc
            try:
                res = self.requests_session.post(upload_url, headers=upload_headers, files=files, timeout=20, allow_redirects=False)
            except requests.exceptions.Timeout:
                return (False, 'Request timeout!')
            except requests.exceptions.ConnectionError:
                return (False, 'Connection error!')

            if res.status_code != 302:
                return (False, 'Unknown error!')

            redirect_url = res.headers.get('Location')
            if not redirect_url:
                return (False, 'Error getting redirect URL!')
            if not redirect_url.startswith('https://'):
                break
            upload_url = redirect_url

        parsed_url = urlparse(redirect_url)
        query_params = parse_qs(parsed_url.query)

        image_insights_token = query_params.get('insightsToken')
        if not image_insights_token:
            return (False, 'Error getting token!')
        image_insights_token = image_insights_token[0]

        api_url = f'https://{api_host}/images/api/custom/knowledge'
        api_headers = {
            'accept': '*/*',
            'accept-language': 'ja-JP;q=0.6,ja;q=0.5',
            'origin': 'https://www.bing.com',
            'referer': f'https://www.bing.com/images/search?view=detailV2&insightstoken={image_insights_token}',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:136.0) Gecko/20100101 Firefox/136.0',
        }
        api_data_json = {
            'imageInfo': {'imageInsightsToken': image_insights_token, 'source': 'Url'},
            'knowledgeRequest': {'invokedSkills': ['OCR'], 'index': 1}
        }
        files = {   
            'knowledgeRequest': (None, json.dumps(api_data_json), 'application/json')
        }

        try:
            res = self.requests_session.post(api_url, headers=api_headers, files=files, timeout=20)
        except requests.exceptions.Timeout:
            return (False, 'Request timeout!')
        except requests.exceptions.ConnectionError:
            return (False, 'Connection error!')

        if res.status_code != 200:
            return (False, 'Unknown error!')

        data = res.json()
        
        ocr_result = self._to_generic_result(data, img.width, img.height)
        x = (True, ocr_result)

        if is_path:
            img.close()
        return x

    def _preprocess(self, img):
        max_pixel_size = 4000
        max_byte_size = 767772
        res = None

        if any(x > max_pixel_size for x in img.size):
            resize_factor = min(max_pixel_size / img.width, max_pixel_size / img.height)
            new_w = int(img.width * resize_factor)
            new_h = int(img.height * resize_factor)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        img_bytes, _ = limit_image_size(img, max_byte_size)

        if img_bytes:
            res = base64.b64encode(img_bytes).decode('utf-8')

        return res

class AppleVision:
    name = 'avision'
    readable_name = 'Apple Vision'
    key = 'a'
    available = False
    local = True
    manual_language = True
    coordinate_support = False

    def __init__(self, language='ja'):
        if sys.platform != 'darwin':
            logger.warning('Apple Vision is not supported on non-macOS platforms!')
        elif int(platform.mac_ver()[0].split('.')[0]) < 13:
            logger.warning('Apple Vision is not supported on macOS older than Ventura/13.0!')
        else:
            self.available = True
            self.language = [language, 'en']
            logger.info('Apple Vision ready')

    def __call__(self, img):
        img, is_path = input_to_pil_image(img)
        if not img:
            return (False, 'Invalid image provided')

        with objc.autorelease_pool():
            req = Vision.VNRecognizeTextRequest.alloc().init()

            req.setRevision_(Vision.VNRecognizeTextRequestRevision3)
            req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
            req.setUsesLanguageCorrection_(True)
            req.setRecognitionLanguages_(self.language)

            handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(
                self._preprocess(img), None
            )

            success = handler.performRequests_error_([req], None)
            res = []
            if success[0]:
                for result in req.results():
                    res.append(result.text())
                x = (True, res)
            else:
                x = (False, 'Unknown error!')

            if is_path:
                img.close()
            return x

    def _preprocess(self, img):
        return pil_image_to_bytes(img, 'tiff')

class AppleLiveText:
    name = 'alivetext'
    readable_name = 'Apple Live Text'
    key = 'd'
    available = False
    local = True
    manual_language = True
    coordinate_support = True

    def __init__(self, language='ja'):
        if sys.platform != 'darwin':
            logger.warning('Apple Live Text is not supported on non-macOS platforms!')
        elif int(platform.mac_ver()[0].split('.')[0]) < 13:
            logger.warning('Apple Live Text is not supported on macOS older than Ventura/13.0!')
        else:
            app_info = NSBundle.mainBundle().infoDictionary()
            app_info['LSBackgroundOnly'] = '1'
            self.VKCImageAnalyzer = objc.lookUpClass('VKCImageAnalyzer')
            self.VKCImageAnalyzerRequest = objc.lookUpClass('VKCImageAnalyzerRequest')
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
            self.language = [language, 'en']
            self.available = True
            logger.info('Apple Live Text ready')

    def __call__(self, img):
        img, is_path = input_to_pil_image(img)
        if not img:
            return (False, 'Invalid image provided')

        self.result = None

        with objc.autorelease_pool():
            analyzer = self.VKCImageAnalyzer.alloc().init()
            req = self.VKCImageAnalyzerRequest.alloc().initWithImage_requestType_(self._preprocess(img), 1) #VKAnalysisTypeText
            req.setLocales_(self.language)
            analyzer.processRequest_progressHandler_completionHandler_(req, lambda progress: None, self._process)

            CFRunLoopRunInMode(kCFRunLoopDefaultMode, 10.0, False)

        if self.result == None:
            return (False, 'Unknown error!')

        ocr_response = OcrResult(
            image_properties=ImageProperties(width=img.width, height=img.height),
            paragraphs=self.result
        )
        x = (True, ocr_response)

        if is_path:
            img.close()
        return x

    def _process(self, analysis, error):
        lines = []
        response_lines = analysis.allLines()
        if response_lines:
            for l in response_lines:
                words = []
                for i, w in enumerate(l.children()):
                    w_bbox = w.quad().boundingBox()
                    word = Word(
                        text=w.string(),
                        bounding_box=BoundingBox(
                            width=w_bbox.size.width,
                            height=w_bbox.size.height,
                            center_x=w_bbox.origin.x + (w_bbox.size.width / 2),
                            center_y=w_bbox.origin.y + (w_bbox.size.height / 2),
                            rotation_z=0.0
                        )
                    )
                    words.append(word)
                
                l_bbox = l.quad().boundingBox()
                line = Line(
                    text=l.string(),
                    bounding_box=BoundingBox(
                        width=l_bbox.size.width,
                        height=l_bbox.size.height,
                        center_x=l_bbox.origin.x + (l_bbox.size.width / 2),
                        center_y=l_bbox.origin.y + (l_bbox.size.height / 2),
                        rotation_z=0.0
                    ),
                    words=words
                )
                lines.append(line)

        # Create a single paragraph to hold all lines
        if lines:
            # Approximate paragraph bbox by combining all line bboxes
            all_line_bboxes = [l.bounding_box for l in lines]
            min_x = min(b.center_x - b.width / 2 for b in all_line_bboxes)
            max_x = max(b.center_x + b.width / 2 for b in all_line_bboxes)
            min_y = min(b.center_y - b.height / 2 for b in all_line_bboxes)
            max_y = max(b.center_y + b.height / 2 for b in all_line_bboxes)
            
            p_bbox = BoundingBox(
                center_x=(min_x + max_x) / 2,
                center_y=(min_y + max_y) / 2,
                width=max_x - min_x,
                height=max_y - min_y
            )
            paragraph = Paragraph(bounding_box=p_bbox, lines=lines)
            paragraphs = [paragraph]
        else:
            paragraphs = []

        self.result = paragraphs
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
    local = True
    manual_language = True
    coordinate_support = False

    def __init__(self, config={}, language='ja'):
        if sys.platform == 'win32':
            if int(platform.release()) < 10:
                logger.warning('WinRT OCR is not supported on Windows older than 10!')
            elif 'winocr' not in sys.modules:
                logger.warning('winocr not available, WinRT OCR will not work!')
            else:
                self.language = language
                self.available = True
                logger.info('WinRT OCR ready')
        else:
            try:
                self.url = config['url']
                self.language = language
                self.available = True
                logger.info('WinRT OCR ready')
            except:
                logger.warning('Error reading URL from config, WinRT OCR will not work!')

    def __call__(self, img):
        img, is_path = input_to_pil_image(img)
        if not img:
            return (False, 'Invalid image provided')

        if sys.platform == 'win32':
            res = winocr.recognize_pil_sync(img, lang=self.language)['text']
        else:
            params = {'lang': self.language}
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

        if is_path:
            img.close()
        return x

    def _preprocess(self, img):
        return pil_image_to_bytes(img, png_compression=1)

class OneOCR:
    name = 'oneocr'
    readable_name = 'OneOCR'
    key = 'z'
    available = False
    local = True
    manual_language = False
    coordinate_support = True

    def __init__(self, config={}):
        if sys.platform == 'win32':
            if int(platform.release()) < 10:
                logger.warning('OneOCR is not supported on Windows older than 10!')
            elif 'oneocr' not in sys.modules:
                logger.warning('oneocr not available, OneOCR will not work!')
            else:
                try:
                    self.model = oneocr.OcrEngine()
                except RuntimeError as e:
                    logger.warning(e + ', OneOCR will not work!')
                else:
                    self.available = True
                    logger.info('OneOCR ready')
        else:
            try:
                self.url = config['url']
                self.available = True
                logger.info('OneOCR ready')
            except:
                logger.warning('Error reading URL from config, OneOCR will not work!')

    def _pixel_quad_to_center_bbox(self, rect, img_width, img_height):
        x_coords = [rect['x1'], rect['x2'], rect['x3'], rect['x4']]
        y_coords = [rect['y1'], rect['y2'], rect['y3'], rect['y4']]

        center_x_px = sum(x_coords) / 4
        center_y_px = sum(y_coords) / 4
        
        width_px = (abs(rect['x2'] - rect['x1']) + abs(rect['x3'] - rect['x4'])) / 2
        height_px = (abs(rect['y4'] - rect['y1']) + abs(rect['y3'] - rect['y2'])) / 2

        return BoundingBox(
            center_x=center_x_px / img_width,
            center_y=center_y_px / img_height,
            width=width_px / img_width,
            height=height_px / img_height
        )

    def _to_generic_result(self, response, img_width, img_height):
        lines = []
        for l in response.get('lines', []):
            words = []
            for i, w in enumerate(l.get('words', [])):
                word = Word(
                    text=w.get('text', ''),
                    bounding_box=self._pixel_quad_to_center_bbox(w['bounding_rect'], img_width, img_height)
                )
                words.append(word)
            
            line = Line(
                text=l.get('text', ''),
                bounding_box=self._pixel_quad_to_center_bbox(l['bounding_rect'], img_width, img_height),
                words=words
            )
            lines.append(line)

        # Create a single paragraph to hold all lines
        if lines:
            # Approximate paragraph bbox by combining all line bboxes
            all_line_bboxes = [l.bounding_box for l in lines]
            min_x = min(b.center_x - b.width / 2 for b in all_line_bboxes)
            max_x = max(b.center_x + b.width / 2 for b in all_line_bboxes)
            min_y = min(b.center_y - b.height / 2 for b in all_line_bboxes)
            max_y = max(b.center_y + b.height / 2 for b in all_line_bboxes)
            
            p_bbox = BoundingBox(
                center_x=(min_x + max_x) / 2,
                center_y=(min_y + max_y) / 2,
                width=max_x - min_x,
                height=max_y - min_y
            )
            paragraph = Paragraph(bounding_box=p_bbox, lines=lines)
            paragraphs = [paragraph]
        else:
            paragraphs = []

        return OcrResult(
            image_properties=ImageProperties(width=img_width, height=img_height),
            paragraphs=paragraphs
        )

    def __call__(self, img):
        img, is_path = input_to_pil_image(img)
        if not img:
            return (False, 'Invalid image provided')

        if sys.platform == 'win32':
            try:
                raw_res = self.model.recognize_pil(img)
            except RuntimeError as e:
                return (False, e)
        else:
            try:
                res = requests.post(self.url, data=self._preprocess(img), timeout=3)
            except requests.exceptions.Timeout:
                return (False, 'Request timeout!')
            except requests.exceptions.ConnectionError:
                return (False, 'Connection error!')

            if res.status_code != 200:
                return (False, 'Unknown error!')

            raw_res = res.json()

        ocr_response = self._to_generic_result(raw_res, img.width, img.height)
        x = (True, ocr_response)

        if is_path:
            img.close()
        return x

    def _preprocess(self, img):
        return pil_image_to_bytes(img, png_compression=1)

class AzureImageAnalysis:
    name = 'azure'
    readable_name = 'Azure Image Analysis'
    key = 'v'
    available = False
    local = False
    manual_language = False
    coordinate_support = False

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

    def __call__(self, img):
        img, is_path = input_to_pil_image(img)
        if not img:
            return (False, 'Invalid image provided')

        try:
            read_result = self.client.analyze(image_data=self._preprocess(img), visual_features=[VisualFeatures.READ])
        except ServiceRequestError:
            return (False, 'Connection error!')
        except:
            return (False, 'Unknown error!')

        res = []
        if read_result.read:
            for block in read_result.read.blocks:
                for line in block.lines:
                    res.append(line.text)
        else:
            return (False, 'Unknown error!')

        x = (True, res)

        if is_path:
            img.close()
        return x

    def _preprocess(self, img):
        if any(x < 50 for x in img.size):
            resize_factor = max(50 / img.width, 50 / img.height)
            new_w = int(img.width * resize_factor)
            new_h = int(img.height * resize_factor)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        return pil_image_to_bytes(img)

class EasyOCR:
    name = 'easyocr'
    readable_name = 'EasyOCR'
    key = 'e'
    available = False
    local = True
    manual_language = True
    coordinate_support = False

    def __init__(self, config={'gpu': True}, language='ja'):
        if 'easyocr' not in sys.modules:
            logger.warning('easyocr not available, EasyOCR will not work!')
        else:
            logger.info('Loading EasyOCR model')
            logging.getLogger('easyocr.easyocr').setLevel(logging.ERROR)
            self.model = easyocr.Reader([language,'en'], gpu=config['gpu'])
            self.available = True
            logger.info('EasyOCR ready')

    def __call__(self, img):
        img, is_path = input_to_pil_image(img)
        if not img:
            return (False, 'Invalid image provided')

        res = []
        read_result = self.model.readtext(self._preprocess(img), detail=0)
        for text in read_result:
            res.append(text)

        x = (True, res)

        if is_path:
            img.close()
        return x

    def _preprocess(self, img):
        return pil_image_to_numpy_array(img)

class RapidOCR:
    name = 'rapidocr'
    readable_name = 'RapidOCR'
    key = 'r'
    available = False
    local = True
    manual_language = True
    coordinate_support = False

    def __init__(self, config={'high_accuracy_detection': False, 'high_accuracy_recognition': True}, language='ja'):
        if 'rapidocr' not in sys.modules:
            logger.warning('rapidocr not available, RapidOCR will not work!')
        else:
            logger.info('Loading RapidOCR model')
            lang_rec = self.language_to_model_language(language)
            self.model = ROCR(params={
                'Det.engine_type': EngineType.ONNXRUNTIME,
                'Det.lang_type': LangDet.CH,
                'Det.model_type': ModelType.SERVER if config['high_accuracy_detection'] else ModelType.MOBILE,
                'Det.ocr_version': OCRVersion.PPOCRV5,
                'Rec.engine_type': EngineType.ONNXRUNTIME,
                'Rec.lang_type': lang_rec,
                'Rec.model_type': ModelType.SERVER if config['high_accuracy_recognition'] else ModelType.MOBILE,
                'Rec.ocr_version': OCRVersion.PPOCRV5,
                'Global.log_level': 'error'
            })
            self.available = True
            logger.info('RapidOCR ready')

    def language_to_model_language(self, language):
        if language == 'ja':
            return LangRec.CH
        if language == 'zh':
            return LangRec.CH
        elif language == 'ko':
            return LangRec.KOREAN
        elif language == 'ru':
            return LangRec.ESLAV
        elif language == 'el':
            return LangRec.EL
        elif language == 'th':
            return LangRec.TH
        else:
            return LangRec.LATIN

    def __call__(self, img):
        img, is_path = input_to_pil_image(img)
        if not img:
            return (False, 'Invalid image provided')

        res = []
        read_results = self.model(self._preprocess(img))
        if read_results:
            for read_result in read_results.txts:
                res.append(read_result)

        x = (True, res)

        if is_path:
            img.close()
        return x

    def _preprocess(self, img):
        return pil_image_to_numpy_array(img)

class OCRSpace:
    name = 'ocrspace'
    readable_name = 'OCRSpace'
    key = 'o'
    available = False
    local = False
    manual_language = True
    coordinate_support = False

    def __init__(self, config={}, language='ja'):
        try:
            self.api_key = config['api_key']
            self.max_byte_size = config.get('file_size_limit', 1000000)
            self.engine_version = config.get('engine_version', 2)
            self.language = self.language_to_model_language(language)
            self.available = True
            logger.info('OCRSpace ready')
        except:
            logger.warning('Error reading API key from config, OCRSpace will not work!')

    def language_to_model_language(self, language):
        if language == 'ja':
            return 'jpn'
        if language == 'zh':
            return 'chs'
        elif language == 'ko':
            return 'kor'
        elif language == 'ar':
            return 'ara'
        elif language == 'ru':
            return 'rus'
        elif language == 'el':
            return 'gre'
        elif language == 'th':
            return 'tha'
        else:
            return 'auto'

    def __call__(self, img):
        img, is_path = input_to_pil_image(img)
        if not img:
            return (False, 'Invalid image provided')

        img_bytes, img_extension = self._preprocess(img)
        if not img_bytes:
            return (False, 'Image is too big!')

        data = {
            'apikey': self.api_key,
            'language': self.language,
            'OCREngine': str(self.engine_version)
        }
        files = {'file': ('image.' + img_extension, img_bytes, 'image/' + img_extension)}

        try:
            res = requests.post('https://api.ocr.space/parse/image', data=data, files=files, timeout=20)
        except requests.exceptions.Timeout:
            return (False, 'Request timeout!')
        except requests.exceptions.ConnectionError:
            return (False, 'Connection error!')

        if res.status_code != 200:
            return (False, 'Unknown error!')

        res = res.json()

        if isinstance(res, str):
            return (False, 'Unknown error!')
        if res['IsErroredOnProcessing']:
            return (False, res['ErrorMessage'])

        res = res['ParsedResults'][0]['ParsedText']
        x = (True, res)

        if is_path:
            img.close()
        return x

    def _preprocess(self, img):       
        return limit_image_size(img, self.max_byte_size)
