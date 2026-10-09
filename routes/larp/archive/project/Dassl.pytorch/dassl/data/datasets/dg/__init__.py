from .pacs import PACS
from .vlcs import VLCS
# WILDS is unrelated to the ten PromptKD datasets and is optional here.
import importlib.util
if importlib.util.find_spec("wilds") is not None:
    from .wilds import *
from .cifar_c import CIFAR10C, CIFAR100C
from .digits_dg import DigitsDG
from .digit_single import DigitSingle
from .office_home_dg import OfficeHomeDG
