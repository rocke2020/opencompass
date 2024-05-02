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
output_root = Path('app/outputs')
back_root = Path('app/summary')


def copy_summaries(model='hf_llama3_8b_instruct'):
    for output in (output_root / model).iterdir():
        if not output.is_dir():
            continue
        summary_dir = output / 'summary'
        if not summary_dir.exists():
            continue
        for summary_csv in summary_dir.glob('*.csv'):
            new_dir = back_root / output.name / 'summary'
            new_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(summary_csv, new_dir)
            logger.info(f'Copied {summary_csv} to {new_dir}')


def main():
    copy_summaries()
    logger.info('end')


if __name__ == '__main__':
    main()
