# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = '8.0.237'

from ultralytics.data.explorer.explorer import Explorer
from ultralytics.models import YOLO
# from ultralytics.models.fastsam import FastSAM
from ultralytics.models.nas import NAS
from ultralytics.utils import SETTINGS as settings
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

__all__ = '__version__', 'YOLO', 'NAS', 'checks', 'download', 'settings', 'Explorer'
