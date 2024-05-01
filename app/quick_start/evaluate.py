import argparse
import json
import math
import os
import random
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from datetime import datetime

from loguru import logger

sys.path.append(os.path.abspath('.'))


def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=2, type=int)
    parser.add_argument('--input_path', type=str, help='')
    _args = parser.parse_args()
    random.seed(_args.seed)
    main(_args)
    logger.info('end')
