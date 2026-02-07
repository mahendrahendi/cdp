"""
Verify ANY CDP image (even new/arbitrary images)
Usage: python verify_any.py <image_path> <template_index>
Example: python verify_any.py phone.jpeg 0  (compares with template 0000.png)
"""

import argparse
import sys
import os
import numpy as np
import yaml
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm

sys.path.insert(0, '..')

import libs.yaml_utils as yaml_utils
from libs.utils import *
import skimage.io
from skimage.color import rgb2gray
from skimage.transform import resize

# ======================================================================================================================
parser = argparse.ArgumentParser(description="Verify any CDP image")
parser.add_argument("image_path", type=str, help="Path to image to verify (any image)")
parser.add_argument("template_index", type=int, help="Index of template to compare against (e.g., 0 for 0000.png)")
parser.add_argument("--config_path", default="./configuration.yml", type=str, help="Config file path")
parser.add_argument("--template_dir", default="../data/binary/", type=str, help="Path to templates")
parser.add_argument("--epoch", default=10, type=int, help="Model checkpoint epoch")
parser.add_argument("--image_type", default="rgb", type=str, help="Image type")
parser.add_argument("--type", default="Dtt_Dxx", type=str, help="Model type")
parser.add_argument("--thr", default=0.5, type=float, help="Binarization threshold")
parser.add_argument("--decision_thr", default=0.0, type=float, help="Decision threshold for OC-SVM (higher = stricter)")

args = parser.parse_args()

# ======================================================================================================================

def _to_array(x):
    while isinstance(x, (list, tuple)) and len(x) == 1:
        x = x[0]
    return np.asarray(x)

def load_and_preprocess_image(image_path, template_path, config):
    """Load and preprocess any image"""

    print(f"[3/4] Loading and preprocessing image...")

    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    if not os.path.exists(template_path):
        print(f"ERROR: Template not found: {template_path}")
        sys.exit(1)

    target_size = config.dataset["args"]["target_size"][0]
    template_size = config.dataset["args"]["template_target_size"][0]

    # Load image
    image_x = skimage.io.imread(image_path).astype(np.float64)

    # Handle different image formats
    if len(image_x.shape) == 2:  # Grayscale
        image_x = np.stack([image_x]*3, axis=-1)  # Convert to RGB
    elif image_x.shape[2] == 4:  # RGBA
        image_x = image_x[:, :, :3]  # Drop alpha

    # Load template
    image_y = skimage.io.imread(template_path).astype(np.float64)
    if len(image_y.shape) == 3:
        image_y = rgb2gray(image_y)

    # Resize to target size (central crop would be better, but resize is simpler for arbitrary images)
    image_x = resize(image_x, (target_size, target_size, 3), preserve_range=True)
    image_y = resize(image_y, (template_size, template_size), preserve_range=True)

    # Normalize using the same method as DataSetLoader
    def normalize_dynamic_range(image):
        """Normalize to [0, 1] range"""
        image_min = np.min(image)
        image_max = np.max(image)
        if image_max - image_min > 0:
            return (image - image_min) / (image_max - image_min)
        else:
            return image

    image_x = normalize_dynamic_range(image_x)
    image_y = normalize_dynamic_range(image_y)

    # Add batch dimension
    x_batch = image_x.reshape((1, target_size, target_size, 3))
    y_batch = image_y.reshape((1, template_size, template_size))

    return x_batch, y_batch


def extract_features(x_batch, y_batch, EstimationModel, config, thr):
    """Extract features from image"""

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
    print("CDP AUTHENTICATION VERIFICATION (ANY IMAGE)")
    print("="*70 + "\n")

    # Load config
    config = yaml_utils.Config(yaml.safe_load(open(args.config_path)))

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
        print(f"ERROR: Model checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    EstimationModel.load_weights(checkpoint_path)

    # Load OC-SVM
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

    # Load and preprocess image
    template_path = os.path.join(args.template_dir, f"{args.template_index:04d}.png")
    x_batch, y_batch = load_and_preprocess_image(args.image_path, template_path, config)

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
    print(f"Image: {args.image_path}")
    print(f"Template: {template_path}")
    print(f"Features: dist_t={features[0]:.2f}, dist_x={features[1]:.6f}")
    print(f"Decision value: {decision_value:.6f}")

    if decision_value >= args.decision_thr:
        print("\n✓ STATUS: AUTHENTIC")
        print("  The image is classified as an ORIGINAL.")
        exit_code = 0
    else:
        print("\n✗ STATUS: FAKE")
        print("  The image is classified as a COUNTERFEIT.")
        exit_code = 1

    print("="*70 + "\n")

    # Show feature comparison
    print("Feature Comparison:")
    print(f"  Training range: dist_t=[{X_train[:,0].min():.0f}, {X_train[:,0].max():.0f}], "
          f"dist_x=[{X_train[:,1].min():.6f}, {X_train[:,1].max():.6f}]")
    print(f"  Your image:     dist_t={features[0]:.0f}, dist_x={features[1]:.6f}")

    if features[0] < X_train[:,0].min() or features[0] > X_train[:,0].max():
        print("  ⚠ Warning: dist_t is outside training range")
    if features[1] < X_train[:,1].min() or features[1] > X_train[:,1].max():
        print("  ⚠ Warning: dist_x is outside training range")

    print()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
