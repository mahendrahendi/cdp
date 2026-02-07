"""
CDP Verification Script - Version 2
Best of both worlds: Works with any image + uses proper preprocessing

Usage:
  python verify_v2.py <image_path> [template_index]

Examples:
  python verify_v2.py phone.jpeg 0              # Compare with template 0 (0000.png)
  python verify_v2.py photo.png                 # Auto-detect template from filename
  python verify_v2.py ../data/original/rgb/0000.png  # Auto-detect (index 0)
  python verify_v2.py my_cdp_photo.jpg 1        # Compare with template 1 (0001.png)
"""

import argparse
import sys
import os
import numpy as np
import yaml
import re
import math
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm

sys.path.insert(0, '..')

import libs.yaml_utils as yaml_utils
from libs.utils import *
import skimage.io
from skimage.color import rgb2gray
from skimage.transform import resize
import scipy.signal

# ======================================================================================================================
parser = argparse.ArgumentParser(description="Verify any CDP image with proper preprocessing")
parser.add_argument("image_path", type=str, help="Path to image (any image, any location)")
parser.add_argument("template_index", type=int, nargs='?', default=None,
                    help="Template index to compare against (optional, auto-detected if not provided)")
parser.add_argument("--config_path", default="./configuration.yml", type=str)
parser.add_argument("--template_dir", default="../data/binary/", type=str)
parser.add_argument("--epoch", default=10, type=int)
parser.add_argument("--image_type", default="rgb", type=str)
parser.add_argument("--type", default="Dtt_Dxx", type=str)
parser.add_argument("--thr", default=0.5, type=float)

args = parser.parse_args()

# ======================================================================================================================

def _to_array(x):
    while isinstance(x, (list, tuple)) and len(x) == 1:
        x = x[0]
    return np.asarray(x)

def extract_template_index(filename):
    """Try to extract template index from filename"""
    match = re.match(r'(\d+)\.png', filename)
    if match:
        return int(match.group(1))
    return None


def load_and_preprocess_image_proper(image_path, template_path, config, symbol_size):
    """
    Load and preprocess image using the SAME method as DataSetLoader
    This ensures consistent features
    """

    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    if not os.path.exists(template_path):
        print(f"ERROR: Template not found: {template_path}")
        sys.exit(1)

    target_size = config.dataset["args"]["target_size"]
    template_target_size = config.dataset["args"]["template_target_size"]

    # Load image (same as DataSetLoader)
    image_x = skimage.io.imread(image_path).astype(np.float64)

    # Handle different formats
    if len(image_x.shape) < len(target_size):
        image_x = image_x.reshape((image_x.shape[0], image_x.shape[1], 1))
    if len(image_x.shape) == 3 and image_x.shape[2] == 4:  # RGBA
        image_x = image_x[:, :, :3]

    # Load template
    image_y = skimage.io.imread(template_path).astype(np.float64)
    if len(image_y.shape) == 3:
        image_y = rgb2gray(image_y)

    # Central crop (same as DataSetLoader)
    def central_crop(image, target_size, symbol_size):
        if image.shape[0] <= target_size[0] and image.shape[1] <= target_size[1]:
            return image

        height, width = image.shape[0:2]
        top_corner = symbol_size * math.floor((height // 2 - target_size[0] // 2) / symbol_size)
        left_corner = symbol_size * math.floor((width // 2 - target_size[1] // 2) / symbol_size)

        return image[top_corner:top_corner+target_size[0],
                    left_corner:left_corner+target_size[1]].reshape(target_size)

    image_x = central_crop(image_x, target_size, symbol_size)
    image_y = central_crop(image_y, template_target_size, symbol_size)

    # Normalize dynamic range (same as DataSetLoader)
    def normalise_dynamic_range(image):
        image_min = np.min(image)
        image_max = np.max(image)
        if image_max - image_min > 0:
            return (image - image_min) / (image_max - image_min)
        return image

    image_x = normalise_dynamic_range(image_x)
    image_y = normalise_dynamic_range(image_y)

    # Add batch dimension
    x_batch = image_x.reshape((1, *target_size))
    y_batch = image_y.reshape((1, *template_target_size))

    return x_batch, y_batch


def extract_features(x_batch, y_batch, EstimationModel, config, thr):
    """Extract features (same as test script)"""

    symbol_size = config.dataset["args"]["symbol_size"]
    target_size = config.dataset["args"]["target_size"][0]
    template_size = config.dataset["args"]["template_target_size"][0]

    # Predict
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
    t_predict_binary = postProcessingSimbolWise(np.copy(t_predict), symbol_size=symbol_size, thr=thr)

    # Calculate distances
    dist_t = np.sum(np.logical_xor(y_sample.reshape((-1)), t_predict_binary.reshape((-1)))) / (symbol_size**2)
    dist_x = mse1D(x_sample.reshape((-1)), x_predict.reshape((-1)))

    return [dist_t, dist_x]


def main():
    print("\n" + "="*70)
    print("CDP AUTHENTICATION VERIFICATION v2")
    print("="*70 + "\n")

    # Auto-detect template index if not provided
    template_index = args.template_index
    if template_index is None:
        filename = os.path.basename(args.image_path)
        template_index = extract_template_index(filename)

        if template_index is None:
            print("Could not auto-detect template index from filename.")
            print(f"Filename: {filename}")
            print("\nPlease specify template index:")
            print("  python verify_v2.py <image> <template_index>")
            print("\nAvailable templates:")
            for i in range(10):
                template_file = os.path.join(args.template_dir, f"{i:04d}.png")
                if os.path.exists(template_file):
                    print(f"  {i} → {i:04d}.png")
            sys.exit(1)
        else:
            print(f"Auto-detected template index: {template_index} (from filename)")

    # Load config
    config = yaml_utils.Config(yaml.safe_load(open(args.config_path)))
    symbol_size = config.dataset["args"]["symbol_size"]

    # Setup
    from libs.EstimatiorModel import TemplateEstimatior
    args.checkpoint_dir = "%s_%s" % (args.image_type, args.type)
    args.dir = "%s" % args.type

    # Load model
    print("[1/4] Loading trained model...")
    Estimator = TemplateEstimatior(config, args, type=args.type)
    EstimationModel = Estimator.EstimationModel

    checkpoint_path = "%s/EstimationModel_epoch_%d.weights.h5" % (Estimator.checkpoint_dir, args.epoch)
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Model checkpoint not found")
        sys.exit(1)

    EstimationModel.load_weights(checkpoint_path)

    # Load OC-SVM
    print("[2/4] Training OC-SVM...")
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

    # Load and preprocess
    print(f"[3/4] Loading image: {os.path.basename(args.image_path)}")
    template_path = os.path.join(args.template_dir, f"{template_index:04d}.png")
    x_batch, y_batch = load_and_preprocess_image_proper(args.image_path, template_path, config, symbol_size)

    # Extract features
    print("[4/4] Extracting features and classifying...")
    features = extract_features(x_batch, y_batch, EstimationModel, config, args.thr)

    # Classify
    X_test = np.array([features])
    prediction = clf.predict(X_test)[0]
    decision_value = clf.decision_function(X_test)[0]

    # Display results
    print("\n" + "="*70)
    print("VERIFICATION RESULT")
    print("="*70)
    print(f"Image:    {args.image_path}")
    print(f"Template: {template_path}")
    print(f"\nFeatures:")
    print(f"  dist_t (Hamming): {features[0]:.2f}")
    print(f"  dist_x (MSE):     {features[1]:.6f}")
    print(f"  Decision value:   {decision_value:.6f}")

    print(f"\nTraining Range:")
    print(f"  dist_t: [{X_train[:,0].min():.0f}, {X_train[:,0].max():.0f}]")
    print(f"  dist_x: [{X_train[:,1].min():.6f}, {X_train[:,1].max():.6f}]")

    # Check if in range
    in_range_t = X_train[:,0].min() <= features[0] <= X_train[:,0].max()
    in_range_x = X_train[:,1].min() <= features[1] <= X_train[:,1].max()

    print(f"\nFeature Analysis:")
    print(f"  dist_t in range: {'✓ Yes' if in_range_t else '✗ No (outside training range!)'}")
    print(f"  dist_x in range: {'✓ Yes' if in_range_x else '✗ No (outside training range!)'}")

    # Final result
    if prediction == 1:
        print("\n" + "="*70)
        print("✓ STATUS: AUTHENTIC")
        print("  The image is classified as an ORIGINAL.")
        print("="*70 + "\n")
        exit_code = 0
    else:
        print("\n" + "="*70)
        print("✗ STATUS: FAKE")
        print("  The image is classified as a COUNTERFEIT.")
        if not in_range_t or not in_range_x:
            print("\n  Note: Features are outside training range.")
            print("  This could mean:")
            print("    - Different printing method")
            print("    - Different capture conditions")
            print("    - Actually a fake/copy")
        print("="*70 + "\n")
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
