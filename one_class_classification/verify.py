"""
Simple CDP Authentication Verification Script
Usage: python verify.py <image_path>
Example: python verify.py ../data/original/rgb/0000.png
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
from libs.DataSetLoader import DataSetLoader
from libs.EstimatiorModel import TemplateEstimatior

# ======================================================================================================================
parser = argparse.ArgumentParser(description="Verify a single CDP image")
parser.add_argument("image_path", type=str, help="Path to the image to verify")
parser.add_argument("--config_path", default="./configuration.yml", type=str, help="Config file path")
parser.add_argument("--template_path", default="../data/binary/", type=str, help="Path to digital templates")
parser.add_argument("--epoch", default=10, type=int, help="Model checkpoint epoch")
parser.add_argument("--image_type", default="rgb", type=str, help="Image type")
parser.add_argument("--type", default="Dtt_Dxx", type=str, help="Model type")
parser.add_argument("--thr", default=0.5, type=float, help="Binarization threshold")

args = parser.parse_args()

# ======================================================================================================================

def load_model_and_ocsvm(config, args):
    """Load trained model and OC-SVM"""

    # Load feature extraction model
    print("[1/4] Loading trained model...")
    args.checkpoint_dir = "%s_%s" % (args.image_type, args.type)
    args.dir = "%s" % args.type
    Estimator = TemplateEstimatior(config, args, type=args.type)
    EstimationModel = Estimator.EstimationModel

    checkpoint_path = "%s/EstimationModel_epoch_%d" % (Estimator.checkpoint_dir, args.epoch)
    if not os.path.exists(checkpoint_path + ".index"):
        print(f"ERROR: Model checkpoint not found at {checkpoint_path}")
        print("Please train the model first: python train_Dtt_Dxx.py --epochs 10")
        sys.exit(1)

    EstimationModel.load_weights(checkpoint_path)

    # Load training features and train OC-SVM
    print("[2/4] Training OC-SVM from saved features...")
    result_dir = "results/%s" % args.type
    file_suf = "%s_%s" % (args.image_type, args.type)
    file_train = "./%s/train_%s.txt" % (result_dir, file_suf)

    if not os.path.exists(file_train):
        print(f"ERROR: Training features not found at {file_train}")
        print("Please extract features first: python test_Dtt_Dxx.py --epoch 10 --subset train --data_paths '../data/original/rgb/'")
        sys.exit(1)

    Dists = loadListFromJson(file_train)
    X_train = np.asarray(Dists[0])

    clf = make_pipeline(StandardScaler(), svm.OneClassSVM(kernel="rbf", nu=0.0005, gamma=0.1))
    clf.fit(X_train)

    return EstimationModel, clf, config


def extract_features_from_image(image_path, template_path, EstimationModel, config, args):
    """Extract features from a single image"""

    print(f"[3/4] Extracting features from: {image_path}")

    if not os.path.exists(image_path):
        print(f"ERROR: Image not found at {image_path}")
        sys.exit(1)

    # Load image
    import skimage.io
    from skimage.color import rgb2gray

    symbol_size = config.dataset["args"]["symbol_size"]
    target_size = config.dataset["args"]["target_size"][0]
    template_size = config.dataset["args"]["template_target_size"][0]

    # Read and preprocess image
    image_x = skimage.io.imread(image_path).astype(np.float64)
    if image_x.shape[2] == 4:  # RGBA
        image_x = image_x[:, :, :3]  # Drop alpha channel

    # Resize to target size
    from skimage.transform import resize
    image_x = resize(image_x, (target_size, target_size, 3), preserve_range=True)

    # Find corresponding template (assume same filename in template directory)
    image_filename = os.path.basename(image_path)
    template_file = os.path.join(template_path, image_filename)

    if not os.path.exists(template_file):
        # Try with .png extension
        template_file = os.path.join(template_path, os.path.splitext(image_filename)[0] + '.png')

    if not os.path.exists(template_file):
        print(f"ERROR: Corresponding template not found at {template_file}")
        print(f"Expected template filename: {image_filename} or {os.path.splitext(image_filename)[0]}.png")
        sys.exit(1)

    # Read template
    image_y = skimage.io.imread(template_file).astype(np.float64)
    if len(image_y.shape) > 2:
        image_y = rgb2gray(image_y)

    # Resize template
    image_y = resize(image_y, (template_size, template_size), preserve_range=True)

    # Normalize
    image_x = image_x / 255.0
    image_y = image_y / 255.0

    # Add batch dimension
    x_batch = image_x.reshape((1, target_size, target_size, 3))
    y_batch = image_y.reshape((1, template_size, template_size))

    # Predict
    prediction = EstimationModel.predict(x_batch)
    t_predict_batch = prediction[0]
    x_predict_batch = prediction[1]

    # Calculate crop size
    crop_size_t = (template_size // symbol_size) * symbol_size
    crop_size_x = (target_size // symbol_size) * symbol_size

    # Process
    t_predict = t_predict_batch[0].reshape((template_size, template_size))[:crop_size_t, :crop_size_t]
    x_predict = x_predict_batch[0].reshape((target_size, target_size, -1))[:crop_size_x, :crop_size_x]
    y_sample = y_batch[0].reshape((template_size, template_size))[:crop_size_t, :crop_size_t]
    x_sample = x_batch[0].reshape((target_size, target_size, -1))[:crop_size_x, :crop_size_x]

    # Binarize prediction
    t_predict_binary = postProcessingSimbolWise(np.copy(t_predict), symbol_size=symbol_size, thr=args.thr)

    # Calculate distances
    dist_t = np.sum(np.logical_xor(y_sample.reshape((-1)), t_predict_binary.reshape((-1)))) / (symbol_size**2)
    dist_x = mse1D(x_sample.reshape((-1)), x_predict.reshape((-1)))

    return [dist_t, dist_x]


def verify_image(features, clf):
    """Classify image as authentic or fake"""

    print("[4/4] Classifying...")

    X_test = np.array([features])
    prediction = clf.predict(X_test)[0]
    decision_value = clf.decision_function(X_test)[0]

    return prediction, decision_value, features


# ======================================================================================================================

def main():
    print("\n" + "="*70)
    print("CDP AUTHENTICATION VERIFICATION")
    print("="*70 + "\n")

    # Load config
    config = yaml_utils.Config(yaml.load(open(args.config_path), Loader=yaml.FullLoader))

    # Load model and OC-SVM
    EstimationModel, clf, config = load_model_and_ocsvm(config, args)

    # Extract features
    features = extract_features_from_image(args.image_path, args.template_path, EstimationModel, config, args)

    # Verify
    prediction, decision_value, features = verify_image(features, clf)

    # Display results
    print("\n" + "="*70)
    print("VERIFICATION RESULT")
    print("="*70)
    print(f"Image: {args.image_path}")
    print(f"Features: dist_t={features[0]:.2f}, dist_x={features[1]:.6f}")
    print(f"Decision value: {decision_value:.6f}")

    if prediction == 1:
        print("\n✓ STATUS: AUTHENTIC")
        print("  The image is classified as an ORIGINAL.")
    else:
        print("\n✗ STATUS: FAKE")
        print("  The image is classified as a COUNTERFEIT.")

    print("="*70 + "\n")

    # Return exit code (0 = authentic, 1 = fake)
    sys.exit(0 if prediction == 1 else 1)


# ======================================================================================================================
if __name__ == "__main__":
    main()
