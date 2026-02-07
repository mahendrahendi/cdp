"""
Test script for single template setup
Extracts features using ONE master template for all photos

Usage:
  python test_single_template.py --epoch 50 --subset train --template_index 0
  python test_single_template.py --epoch 50 --subset test --template_index 0
"""

import argparse
import yaml
import sys
import os
import numpy as np

sys.path.insert(0, '..')

import libs.yaml_utils as yaml_utils
from libs.utils import *
from libs.EstimatiorModel import TemplateEstimatior

import skimage.io
from skimage.color import rgb2gray
import math

# ========================================================================================================== 
parser = argparse.ArgumentParser(description="Test with single template")
parser.add_argument("--config_path", default="./configuration.yml", type=str)
parser.add_argument("--template_index", default=0, type=int)
parser.add_argument("--photos_path", default="../data/original/rgb/", type=str)
parser.add_argument("--template_path", default="../data/binary/", type=str)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--subset", default="test", type=str, choices=["train", "test", "validation"])
parser.add_argument("--epoch", default=100, type=int)
parser.add_argument("--image_type", default="rgb", type=str)
parser.add_argument("--type", default="Dtt_Dxx", type=str)
parser.add_argument("--thr", default=0.5, type=float)
parser.add_argument("--is_debug", default=False, type=int)

args = parser.parse_args()

# ==========================================================================================================

def run(args):
    config = yaml_utils.Config(yaml.safe_load(open(args.config_path)))

    symbol_size = config.dataset["args"]["symbol_size"]
    target_size = config.dataset["args"]["target_size"][0]
    template_size = config.dataset["args"]["template_target_size"][0]

    args.checkpoint_dir = "%s_%s" % (args.image_type, args.type)
    args.dir = "%s" % args.type
    args.save_suf = "%s_%s" % (args.image_type, args.type)

    print("Loading model...")
    Estimator = TemplateEstimatior(config, args, type=args.type)
    EstimationModel = Estimator.EstimationModel

    EstimationModel.load_weights("%s/EstimationModel_epoch_%d.weights.h5" % (Estimator.checkpoint_dir, args.epoch))

    # Note: For now, still use the standard test script with duplicated templates
    # Or use verify_v2.py for individual verification
    print("\nNOTE: For feature extraction with single template,")
    print("please use verify_v2.py for individual image verification.")
    print("\nFor batch testing, temporarily duplicate templates using:")
    print("  python duplicate_templates.py")
    print("Then use the standard test scripts.")

if __name__ == "__main__":
    run(args)
