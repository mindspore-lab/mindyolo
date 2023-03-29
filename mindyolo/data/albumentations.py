import platform
import random
import numpy as np
import pkg_resources as pkg


class Albumentations:
    # Implement Albumentations augmentation https://github.com/ultralytics/yolov5
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self, size=640):
        self.transform = None
        prefix = _colorstr('albumentations: ')
        try:
            import albumentations as A
            _check_version(A.__version__, '1.0.3', hard=True)  # version requirement
            T = [
                A.RandomResizedCrop(height=size, width=size, scale=(0.8, 1.0), ratio=(0.9, 1.11), p=0.0),
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0)]  # transforms
            self.transform = A.Compose(T, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

            print(prefix + ', '.join(f'{x}'.replace('always_apply=False, ', '') for x in T if x.p), flush=True)
            print("[INFO] albumentations load success", flush=True)
        except ImportError:  # package not installed, skip
            pass
            print("[WARNING] package not installed, albumentations load failed", flush=True)
        except Exception as e:
            print(f'{prefix}{e}', flush=True)
            print("[WARNING] albumentations load failed", flush=True)

    def __call__(self, im, labels, p=1.0):
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
            im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])
        return im, labels


def _check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False, verbose=False):
        # Check version vs. required version
        current, minimum = (pkg.parse_version(x) for x in (current, minimum))
        result = (current == minimum) if pinned else (current >= minimum)  # bool
        s = f'WARNING ⚠️ {name}{minimum} is required by YOLOv5, but {name}{current} is currently installed'  # string
        if hard:
            assert result, _emojis(s)  # assert min requirements met
        if verbose and not result:
            print(s, flush=True)
        return result


def _emojis(string=''):
        # Return platform-dependent emoji-safe version of string
        return string.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else string


def _colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']
