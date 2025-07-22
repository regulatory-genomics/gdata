from . import utils
from gdata._gdata import GenomeDataBuilder, GenomeDataLoader, GenomeDataLoaderMap, __version__

import sys
import logging

logging.basicConfig(
    stream=sys.stderr,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO, 
)