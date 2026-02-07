"""
Training script for single template with multiple photos
Use Case: Many photos of the SAME CDP (one master template)

Usage:
  python train_single_template.py --epochs 50 --template_index 0
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

# Custom DataSetLoader that uses single template
import skimage.io
from skimage.color import rgb2gray
from skimage.transform import resize
import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ======================================================================================================================
parser = argparse.ArgumentParser(description="Train with single template, multiple photos")
parser.add_argument("--config_path", default="./configuration.yml", type=str)
parser.add_argument("--template_index", default=0, type=int, help="Master template index")
parser.add_argument("--photos_path", default="../data/original/rgb/", type=str)
parser.add_argument("--template_path", default="../data/binary/", type=str)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--image_type", default="rgb", type=str)
parser.add_argument("--type", default="Dtt_Dxx", type=str)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--start_epoch", default=0, type=int)
parser.add_argument("--is_debug", default=True, type=int)

args = parser.parse_args()
# ======================================================================================================================

def load_single_template_dataset(config, args):
    """Load dataset with SINGLE master template for ALL photos"""

    print(f"Loading photos from: {args.photos_path}")
    print(f"Using master template: {args.template_index}")

    # Get all photo files
    photo_files = sorted([f for f in os.listdir(args.photos_path) if f.endswith('.png')])

    if len(photo_files) == 0:
        print(f"ERROR: No photos found in {args.photos_path}")
        sys.exit(1)

    print(f"Found {len(photo_files)} photos")

    # Load master template
    template_file = os.path.join(args.template_path, f"{args.template_index:04d}.png")
    if not os.path.exists(template_file):
        print(f"ERROR: Master template not found: {template_file}")
        sys.exit(1)

    print(f"Master template: {template_file}")

    # Load and preprocess data
    target_size = config.dataset["args"]["target_size"]
    template_target_size = config.dataset["args"]["template_target_size"]
    symbol_size = config.dataset["args"]["symbol_size"]

    photos = []
    templates = []

    # Load master template once
    master_template = skimage.io.imread(template_file).astype(np.float64)
    if len(master_template.shape) == 3:
        master_template = rgb2gray(master_template)

    # Central crop helper
    def central_crop(image, target_size_list, symbol_size):
        if image.shape[0] <= target_size_list[0] and image.shape[1] <= target_size_list[1]:
            return image

        height, width = image.shape[0:2]
        top_corner = symbol_size * math.floor((height // 2 - target_size_list[0] // 2) / symbol_size)
        left_corner = symbol_size * math.floor((width // 2 - target_size_list[1] // 2) / symbol_size)

        return image[top_corner:top_corner+target_size_list[0],
                    left_corner:left_corner+target_size_list[1]].reshape(target_size_list)

    # Normalize helper
    def normalise_dynamic_range(image):
        image_min = np.min(image)
        image_max = np.max(image)
        if image_max - image_min > 0:
            return (image - image_min) / (image_max - image_min)
        return image

    # Process master template
    master_template = central_crop(master_template, template_target_size, symbol_size)
    master_template = normalise_dynamic_range(master_template)

    # Load all photos
    for photo_file in photo_files:
        photo_path = os.path.join(args.photos_path, photo_file)

        # Load photo
        photo = skimage.io.imread(photo_path).astype(np.float64)
        if len(photo.shape) < len(target_size):
            photo = photo.reshape((photo.shape[0], photo.shape[1], 1))
        if len(photo.shape) == 3 and photo.shape[2] == 4:  # RGBA
            photo = photo[:, :, :3]

        # Preprocess
        photo = central_crop(photo, target_size, symbol_size)
        photo = normalise_dynamic_range(photo)

        photos.append(photo)
        templates.append(master_template)  # Use SAME master template for all!

    photos = np.array(photos)
    templates = np.array(templates)

    print(f"Dataset loaded: {len(photos)} photos, all using same template")

    # Split train/val/test
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(photos))
    train_idx, test_idx = train_test_split(indices,
                                            test_size=config.dataset["args"]["test_ration"],
                                            random_state=args.seed)
    train_idx, val_idx = train_test_split(train_idx,
                                          test_size=config.dataset["args"]["val_ratio"],
                                          random_state=args.seed)

    print(f"Split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # Return train data (we'll implement data generator below)
    return photos[train_idx], templates[train_idx], (photos, templates, train_idx, val_idx, test_idx)


def create_data_generator(x_data, y_data, config):
    """Create data generator with augmentation"""

    batch_size = config.batchsize
    aug_config = config.dataset["args"]["augmentation_args"]

    # Apply augmentations
    augmented_x = []
    augmented_y = []

    for x, y in zip(x_data, y_data):
        # Original
        augmented_x.append(x)
        augmented_y.append(y)

        # Rotations
        for angle in aug_config["rotation_angles"][1:]:  # Skip 0 degrees
            from skimage.transform import rotate as sk_rotate
            x_rot = sk_rotate(x, angle, preserve_range=True)
            augmented_x.append(x_rot)
            augmented_y.append(y)  # Template doesn't rotate

        # Gamma corrections
        from skimage.exposure import adjust_gamma
        gamma_range = aug_config["gamma"]
        for gamma in np.arange(gamma_range[0], gamma_range[1], gamma_range[2]):
            if gamma == 1.0:
                continue
            x_gamma = adjust_gamma(x, gamma)
            augmented_x.append(x_gamma)
            augmented_y.append(y)

    augmented_x = np.array(augmented_x)
    augmented_y = np.array(augmented_y)

    print(f"Data augmentation: {len(x_data)} → {len(augmented_x)} samples")

    # Create generator
    datagen = ImageDataGenerator()

    return datagen.flow(augmented_x, augmented_y, batch_size=batch_size)


def train(args):
    config = yaml_utils.Config(yaml.safe_load(open(args.config_path)))

    args.checkpoint_dir = "%s_%s" % (args.image_type, args.type)
    args.dir = "%s" % args.type

    print("="*70)
    print("TRAINING WITH SINGLE TEMPLATE")
    print("="*70)

    # Load data
    x_train, y_train, full_data = load_single_template_dataset(config, args)

    # Create model
    print("\nInitializing model...")
    Estimator = TemplateEstimatior(config, args, type=args.type)
    EstimationModel = Estimator.EstimationModel

    if args.is_debug:
        print("\nModel architecture:")
        Estimator.UnetXModel.summary()

    # Load checkpoint if continuing
    if args.start_epoch > 0:
        checkpoint_path = "%s/EstimationModel_epoch_%d.weights.h5" % (Estimator.checkpoint_dir, args.start_epoch)
        print(f"\nLoading checkpoint from epoch {args.start_epoch}...")
        EstimationModel.load_weights(checkpoint_path)

    # Create data generator
    print("\nPreparing data with augmentation...")
    train_gen = create_data_generator(x_train, y_train, config)

    # Training
    print("\n" + "="*70)
    print("TRAINING START")
    print("="*70 + "\n")

    n_batches = len(x_train) // config.batchsize

    for epoch in range(args.start_epoch + 1, args.epochs + 1):
        Loss_x = []
        Loss_t = []

        train_gen.reset()

        for batch_idx in range(n_batches * 10):  # Account for augmentation
            x_batch, y_batch = next(train_gen)

            y_batch = y_batch.reshape((-1, config.dataset["args"]["target_size"][0],
                                      config.dataset["args"]["target_size"][1], 1))

            loss = EstimationModel.train_on_batch(x_batch, [y_batch, x_batch])
            Loss_t.append(loss[1])
            Loss_x.append(loss[2])

        print(f"epoch : {epoch:3d}, \t"
              f"mse_t = {np.mean(Loss_t):.6f}\t "
              f"mse_x = {np.mean(Loss_x):.6f}")

        # Save checkpoint
        save_each = saveSpeed(epoch)
        if epoch % save_each == 0 or epoch == args.epochs:
            checkpoint_path = "%s/EstimationModel_epoch_%d.weights.h5" % (Estimator.checkpoint_dir, epoch)
            EstimationModel.save_weights(checkpoint_path)
            print(f"  → Saved checkpoint: {checkpoint_path}")

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nFinal checkpoint: {Estimator.checkpoint_dir}/EstimationModel_epoch_{args.epochs}.weights.h5")
    print(f"\nNext steps:")
    print(f"1. Extract features:")
    print(f"   python test_single_template.py --epoch {args.epochs} --subset train")
    print(f"   python test_single_template.py --epoch {args.epochs} --subset test")
    print(f"2. Train OC-SVM:")
    print(f"   python oc-svm_Dtt_Dxx.py")
    print(f"3. Verify images:")
    print(f"   python verify_v2.py <image> {args.template_index} --epoch {args.epochs}")


# ======================================================================================================================
if __name__ == "__main__":
    set_log_config(args.is_debug)
    log.info("PID = %d\n" % os.getpid())

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    train(args)
