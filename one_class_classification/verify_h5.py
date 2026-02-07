"""
Verify ANY CDP image with robust model loading.

Supports:
1) Modern Keras weights: *.weights.h5 / *.h5
2) Legacy TF checkpoint: prefix + .index + .data-00000-of-00001

Usage:
  python verify_h5.py <image_path> <template_index> --epoch 100
  python verify_h5.py <image_path> <template_index> --model_path checkpoints/rgb_Dtt_Dxx/EstimationModel_epoch_100.weights.h5
  python verify_h5.py <image_path> <template_index> --model_path checkpoints/rgb_Dtt_Dxx/EstimationModel_epoch_100
"""

import argparse
import os
import sys

import numpy as np
import yaml
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "..")

import libs.yaml_utils as yaml_utils
from libs.utils import loadListFromJson, mse1D, postProcessingSimbolWise
import skimage.io
from skimage.color import rgb2gray
from skimage.transform import resize


parser = argparse.ArgumentParser(description="Verify any CDP image")
parser.add_argument("image_path", type=str, help="Path to image to verify")
parser.add_argument("template_index", type=int, help="Template index, e.g. 0 for 0000.png")
parser.add_argument("--config_path", default="./configuration.yml", type=str, help="Config file path")
parser.add_argument("--template_dir", default="../data/binary/", type=str, help="Path to templates")
parser.add_argument("--epoch", default=10, type=int, help="Model checkpoint epoch")
parser.add_argument("--model_path", default="", type=str, help="Optional explicit model path/prefix")
parser.add_argument("--image_type", default="rgb", type=str, help="Image type")
parser.add_argument("--type", default="Dtt_Dxx", type=str, help="Model type")
parser.add_argument("--thr", default=0.5, type=float, help="Binarization threshold")
args = parser.parse_args()


def _to_array(x):
    while isinstance(x, (list, tuple)) and len(x) == 1:
        x = x[0]
    return np.asarray(x)


def _checkpoint_candidates(explicit_model_path, checkpoint_dir, epoch):
    if explicit_model_path:
        p = explicit_model_path
        if p.endswith(".index"):
            p = p[:-len(".index")]
        elif p.endswith(".data-00000-of-00001"):
            p = p[:-len(".data-00000-of-00001")]
        if p.endswith(".weights.h5") or p.endswith(".h5"):
            base = p[:-len(".weights.h5")] if p.endswith(".weights.h5") else p[:-len(".h5")]
            return [p, base, base + ".weights.h5", base + ".h5"]
        # Prefer legacy checkpoint prefix first to avoid bad .weights.h5 collisions.
        return [p, p + ".weights.h5", p + ".h5"]

    base = "%s/EstimationModel_epoch_%d" % (checkpoint_dir, epoch)
    # Prefer legacy checkpoint first in old TF2.0 projects.
    return [base, base + ".weights.h5", base + ".h5"]


def _load_weights_robust(model, explicit_model_path, checkpoint_dir, epoch):
    candidates = _checkpoint_candidates(explicit_model_path, checkpoint_dir, epoch)
    errors = []

    for path in candidates:
        try:
            if path.endswith(".h5"):
                if not os.path.exists(path):
                    continue
                status = model.load_weights(path)
                if hasattr(status, "expect_partial"):
                    status.expect_partial()
                return path
            idx = path + ".index"
            dat = path + ".data-00000-of-00001"
            if not (os.path.exists(idx) and os.path.exists(dat)):
                continue
            # First try Keras checkpoint loader.
            try:
                status = model.load_weights(path)
                if hasattr(status, "expect_partial"):
                    status.expect_partial()
                return path + " (keras checkpoint)"
            except Exception as keras_exc:
                # Fallback for object-based checkpoints created with tf.train.Checkpoint.write().
                try:
                    import tensorflow as tf
                    ckpt = tf.train.Checkpoint(model=model)
                    status = ckpt.restore(path)
                    if hasattr(status, "assert_existing_objects_matched"):
                        status.assert_existing_objects_matched()
                    elif hasattr(status, "expect_partial"):
                        status.expect_partial()
                    return path + " (object checkpoint)"
                except Exception as obj_exc:
                    raise RuntimeError(
                        "keras loader failed: %s | object loader failed: %s"
                        % (
                            str(keras_exc).splitlines()[0] if str(keras_exc) else repr(keras_exc),
                            str(obj_exc).splitlines()[0] if str(obj_exc) else repr(obj_exc),
                        )
                    )
        except Exception as exc:
            first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
            errors.append("%s -> %s: %s" % (path, type(exc).__name__, first_line))

    print("ERROR: Could not load checkpoint.")
    print("Tried paths:")
    for c in candidates:
        print("  - %s" % c)
    if errors:
        print("Load errors:")
        for e in errors:
            print("  - %s" % e)
    sys.exit(1)


def load_and_preprocess_image(image_path, template_path, config):
    print("[3/4] Loading and preprocessing image...")

    if not os.path.exists(image_path):
        print("ERROR: Image not found: %s" % image_path)
        sys.exit(1)
    if not os.path.exists(template_path):
        print("ERROR: Template not found: %s" % template_path)
        sys.exit(1)

    target_size = config.dataset["args"]["target_size"][0]
    template_size = config.dataset["args"]["template_target_size"][0]

    image_x = skimage.io.imread(image_path).astype(np.float64)
    if len(image_x.shape) == 2:
        image_x = np.stack([image_x] * 3, axis=-1)
    elif image_x.shape[2] == 4:
        image_x = image_x[:, :, :3]

    image_y = skimage.io.imread(template_path).astype(np.float64)
    if len(image_y.shape) == 3:
        image_y = rgb2gray(image_y)

    image_x = resize(image_x, (target_size, target_size, 3), preserve_range=True)
    image_y = resize(image_y, (template_size, template_size), preserve_range=True)

    def normalize_dynamic_range(image):
        image_min = np.min(image)
        image_max = np.max(image)
        if image_max - image_min > 0:
            return (image - image_min) / (image_max - image_min)
        return image

    image_x = normalize_dynamic_range(image_x)
    image_y = normalize_dynamic_range(image_y)

    x_batch = image_x.reshape((1, target_size, target_size, 3))
    y_batch = image_y.reshape((1, template_size, template_size))
    return x_batch, y_batch


def extract_features(x_batch, y_batch, estimation_model, config, thr):
    symbol_size = config.dataset["args"]["symbol_size"]
    target_size = config.dataset["args"]["target_size"][0]
    template_size = config.dataset["args"]["template_target_size"][0]

    prediction = estimation_model.predict(x_batch)
    t_predict_batch = _to_array(prediction[0])
    x_predict_batch = _to_array(prediction[1])

    crop_size_t = (template_size // symbol_size) * symbol_size
    crop_size_x = (target_size // symbol_size) * symbol_size

    t_predict = t_predict_batch[0].reshape((template_size, template_size))[:crop_size_t, :crop_size_t]
    x_predict = x_predict_batch[0].reshape((target_size, target_size, -1))[:crop_size_x, :crop_size_x]
    y_sample = y_batch[0].reshape((template_size, template_size))[:crop_size_t, :crop_size_t]
    x_sample = x_batch[0].reshape((target_size, target_size, -1))[:crop_size_x, :crop_size_x]

    t_predict_binary = postProcessingSimbolWise(np.copy(t_predict), symbol_size=symbol_size, thr=thr)

    dist_t = np.sum(np.logical_xor(y_sample.reshape((-1)), t_predict_binary.reshape((-1)))) / (symbol_size ** 2)
    dist_x = mse1D(x_sample.reshape((-1)), x_predict.reshape((-1)))
    return [dist_t, dist_x]


def main():
    print("\n" + "=" * 70)
    print("CDP AUTHENTICATION VERIFICATION (ANY IMAGE)")
    print("=" * 70 + "\n")

    config = yaml_utils.Config(yaml.load(open(args.config_path), Loader=yaml.FullLoader))

    from libs.EstimatiorModel import TemplateEstimatior

    args.checkpoint_dir = "%s_%s" % (args.image_type, args.type)
    args.dir = "%s" % args.type

    print("[1/4] Loading trained model...")
    estimator = TemplateEstimatior(config, args, type=args.type)
    estimation_model = estimator.EstimationModel

    loaded_from = _load_weights_robust(
        estimation_model,
        args.model_path,
        estimator.checkpoint_dir,
        args.epoch,
    )
    print("Loaded model weights from: %s" % loaded_from)

    print("[2/4] Training OC-SVM from saved features...")
    result_dir = "results/%s" % args.type
    file_suf = "%s_%s" % (args.image_type, args.type)
    file_train = "./%s/train_%s.txt" % (result_dir, file_suf)
    if not os.path.exists(file_train):
        print("ERROR: Training features not found: %s" % file_train)
        sys.exit(1)

    dists = loadListFromJson(file_train)
    x_train = np.asarray(dists[0])
    clf = make_pipeline(StandardScaler(), svm.OneClassSVM(kernel="rbf", nu=0.0005, gamma=0.1))
    clf.fit(x_train)

    template_path = os.path.join(args.template_dir, "%04d.png" % args.template_index)
    x_batch, y_batch = load_and_preprocess_image(args.image_path, template_path, config)

    print("[4/4] Extracting features and classifying...")
    features = extract_features(x_batch, y_batch, estimation_model, config, args.thr)

    x_test = np.array([features])
    prediction = clf.predict(x_test)[0]
    decision_value = clf.decision_function(x_test)[0]

    print("\n" + "=" * 70)
    print("VERIFICATION RESULT")
    print("=" * 70)
    print("Image: %s" % args.image_path)
    print("Template: %s" % template_path)
    print("Features: dist_t=%.2f, dist_x=%.6f" % (features[0], features[1]))
    print("Decision value: %.6f" % decision_value)

    if prediction == 1:
        print("\nSTATUS: AUTHENTIC")
        print("The image is classified as an ORIGINAL.")
        exit_code = 0
    else:
        print("\nSTATUS: FAKE")
        print("The image is classified as a COUNTERFEIT.")
        exit_code = 1

    print("=" * 70 + "\n")

    print("Feature Comparison:")
    print(
        "  Training range: dist_t=[%.0f, %.0f], dist_x=[%.6f, %.6f]"
        % (x_train[:, 0].min(), x_train[:, 0].max(), x_train[:, 1].min(), x_train[:, 1].max())
    )
    print("  Your image:     dist_t=%.0f, dist_x=%.6f" % (features[0], features[1]))

    if features[0] < x_train[:, 0].min() or features[0] > x_train[:, 0].max():
        print("  Warning: dist_t is outside training range")
    if features[1] < x_train[:, 1].min() or features[1] > x_train[:, 1].max():
        print("  Warning: dist_x is outside training range")

    print()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
