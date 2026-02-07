"""
Simple CDP Authentication Verification Script
Uses the same preprocessing as test script to ensure consistent results
Usage: python verify_simple.py <image_path>
Example: python verify_simple.py 0000.png
Example: python verify_simple.py ../data/original/rgb/0000.png
"""

import argparse
import sys
import os
import numpy as np
import yaml
import re
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm

sys.path.insert(0, '..')

import libs.yaml_utils as yaml_utils
from libs.utils import *
from libs.DataSetLoader import DataSetLoader
from libs.EstimatiorModel import TemplateEstimatior

# ======================================================================================================================
parser = argparse.ArgumentParser(description="Verify a single CDP image")
parser.add_argument("image_path", type=str, help="Path or filename of image to verify (e.g., 0000.png or ../data/original/rgb/0000.png)")
parser.add_argument("--config_path", default="./configuration.yml", type=str, help="Config file path")
parser.add_argument("--data_path", default="../data/original/rgb/", type=str, help="Path to images directory")
parser.add_argument("--epoch", default=10, type=int, help="Model checkpoint epoch")
parser.add_argument("--image_type", default="rgb", type=str, help="Image type")
parser.add_argument("--type", default="Dtt_Dxx", type=str, help="Model type")
parser.add_argument("--thr", default=0.5, type=float, help="Binarization threshold")

args = parser.parse_args()

# ======================================================================================================================

def _to_array(x):
    while isinstance(x, (list, tuple)) and len(x) == 1:
        x = x[0]
    return np.asarray(x)

def extract_image_index(image_path):
    """Extract image index from filename (e.g., 0000.png -> 0, 0042.png -> 42)"""
    # Get just the filename without path
    filename = os.path.basename(image_path)

    # Extract the numeric part (assumes format like 0000.png, 0001.png, etc.)
    match = re.match(r'(\d+)\.png', filename)
    if match:
        return int(match.group(1))
    else:
        print(f"ERROR: Could not extract index from filename: {filename}")
        print("Expected format: 0000.png, 0001.png, etc.")
        sys.exit(1)

# ======================================================================================================================

def main():
    print("\n" + "="*70)
    print("CDP AUTHENTICATION VERIFICATION")
    print("="*70 + "\n")

    # Extract image index from filename
    image_index = extract_image_index(args.image_path)
    filename = os.path.basename(args.image_path)

    # If full path provided, extract directory
    if os.path.dirname(args.image_path):
        args.data_path = os.path.dirname(args.image_path)

    # Load config
    config = yaml_utils.Config(yaml.safe_load(open(args.config_path)))

    # Setup args
    args.checkpoint_dir = "%s_%s" % (args.image_type, args.type)
    args.dir = "%s" % args.type
    args.printed_path = args.data_path

    symbol_size = config.dataset["args"]["symbol_size"]
    target_size = config.dataset["args"]["target_size"][0]
    template_size = config.dataset["args"]["template_target_size"][0]

    # Load model
    print("[1/4] Loading trained model...")
    Estimator = TemplateEstimatior(config, args, type=args.type)
    EstimationModel = Estimator.EstimationModel

    checkpoint_path = "%s/EstimationModel_epoch_%d.weights.h5" % (Estimator.checkpoint_dir, args.epoch)
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Model checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    EstimationModel.load_weights(checkpoint_path)

    # Load OC-SVM training features
    print("[2/4] Training OC-SVM from saved features...")
    result_dir = "results/%s" % args.type
    file_suf = "%s_%s" % (args.image_type, args.type)
    file_train = "./%s/train_%s.txt" % (result_dir, file_suf)

    if not os.path.exists(file_train):
        print(f"ERROR: Training features not found")
        sys.exit(1)

    Dists = loadListFromJson(file_train)
    X_train = np.asarray(Dists[0])

    clf = make_pipeline(StandardScaler(), svm.OneClassSVM(kernel="rbf", nu=0.0005, gamma=0.1))
    clf.fit(X_train)

    # Load and process image using DataSetLoader (same as test script!)
    print(f"[3/4] Loading image {filename}...")

    # Create temporary config with single image index
    config.dataset["args"]["test_indices"] = [image_index]

    DataGen = DataSetLoader(config, args, type="test", is_debug_mode=False)
    DataGen.initDataSet()

    # Get the image data
    x_data, y_data = DataGen.getData()

    if len(x_data) == 0:
        print(f"ERROR: Image {filename} not found in {args.data_path}")
        sys.exit(1)

    x_batch = x_data[0:1]  # First image as batch
    y_batch = y_data[0:1]

    # Extract features (same as test script!)
    print("[4/4] Extracting features and classifying...")

    prediction = EstimationModel.predict(x_batch)
    t_predict_batch = _to_array(prediction[0])
    x_predict_batch = _to_array(prediction[1])

    # Calculate crop size
    crop_size_t = (template_size // symbol_size) * symbol_size
    crop_size_x = (target_size // symbol_size) * symbol_size

    # Process
    t_predict = t_predict_batch[0].reshape((template_size, template_size))[:crop_size_t, :crop_size_t]
    x_predict = x_predict_batch[0].reshape((target_size, target_size, -1))[:crop_size_x, :crop_size_x]
    y_sample = y_batch[0].reshape((template_size, template_size))[:crop_size_t, :crop_size_t]
    x_sample = x_batch[0].reshape((target_size, target_size, -1))[:crop_size_x, :crop_size_x]

    # Binarize
    t_predict_binary = postProcessingSimbolWise(np.copy(t_predict), symbol_size=symbol_size, thr=args.thr)

    # Calculate distances
    dist_t = np.sum(np.logical_xor(y_sample.reshape((-1)), t_predict_binary.reshape((-1)))) / (symbol_size**2)
    dist_x = mse1D(x_sample.reshape((-1)), x_predict.reshape((-1)))

    # Classify
    X_test = np.array([[dist_t, dist_x]])
    prediction = clf.predict(X_test)[0]
    decision_value = clf.decision_function(X_test)[0]

    # Display results
    print("\n" + "="*70)
    print("VERIFICATION RESULT")
    print("="*70)
    print(f"Image: {filename}")
    print(f"Path: {args.data_path}")
    print(f"Features: dist_t={dist_t:.2f}, dist_x={dist_x:.6f}")
    print(f"Decision value: {decision_value:.6f}")

    if prediction == 1:
        print("\n✓ STATUS: AUTHENTIC")
        print("  The image is classified as an ORIGINAL.")
        exit_code = 0
    else:
        print("\n✗ STATUS: FAKE")
        print("  The image is classified as a COUNTERFEIT.")
        exit_code = 1

    print("="*70 + "\n")

    sys.exit(exit_code)


# ======================================================================================================================
if __name__ == "__main__":
    main()
