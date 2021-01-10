"""Add {PROJECT_ROOT}/lib. to PYTHONPATH
Usage:
import this module before import any modules under lib/
e.g
    import _init_paths
    from core.config import configs
"""

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.abspath(osp.dirname(__file__))
add_path(this_dir)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, 'utils')
add_path(lib_path)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, 'lane_detector')
add_path(lib_path)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, 'lane_detector/lib')
add_path(lib_path)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, 'car_detector')
add_path(lib_path)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, 'car_detector/lib')
add_path(lib_path)