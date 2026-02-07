"""
Convert modern Keras .weights.h5 model weights to legacy TensorFlow checkpoint
files (.index + .data-00000-of-00001) for the one-class CDP model.

Example:
  python one_class_classification/convert_weights_to_legacy_ckpt.py \
    --in_h5 one_class_classification/checkpoints2/rgb_Dtt_Dxx/EstimationModel_epoch_100.weights.h5 \
    --out_prefix one_class_classification/checkpoints/rgb_Dtt_Dxx/EstimationModel_epoch_1
"""

import argparse
from pathlib import Path
import sys

import tensorflow as tf
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

import libs.yaml_utils as yaml_utils
from libs.EstimatiorModel import TemplateEstimatior


def _normalize_output_prefix(out_prefix: Path) -> Path:
    """Allow users to pass either raw prefix or a checkpoint file path."""
    name = out_prefix.name
    for suffix in (".index", ".data-00000-of-00001", ".weights.h5"):
        if name.endswith(suffix):
            return out_prefix.with_name(name[: -len(suffix)])
    return out_prefix


def build_model(config_path: Path, image_type: str, model_type: str):
    config = yaml_utils.Config(yaml.safe_load(config_path.read_text()))
    args = argparse.Namespace(
        checkpoint_dir=f"{image_type}_{model_type}",
        dir=model_type,
        image_type=image_type,
        type=model_type,
    )
    estimator = TemplateEstimatior(config, args, type=model_type)
    return estimator.EstimationModel


def main():
    parser = argparse.ArgumentParser(
        description="Convert .weights.h5 into legacy TF checkpoint files (.index + .data)."
    )
    parser.add_argument(
        "--config_path",
        default=str(SCRIPT_DIR / "configuration.yml"),
        type=str,
        help="Path to one_class_classification/configuration.yml",
    )
    parser.add_argument(
        "--image_type",
        default="rgb",
        choices=["rgb", "gray"],
        help="Model image type used during training",
    )
    parser.add_argument(
        "--type",
        default="Dtt_Dxx",
        choices=["Dtt_Dxx", "Dtt_Dt_Dxx_Dx"],
        help="Model architecture type",
    )
    parser.add_argument(
        "--in_h5",
        required=True,
        type=str,
        help="Input .weights.h5 file",
    )
    parser.add_argument(
        "--out_prefix",
        required=True,
        type=str,
        help="Output checkpoint prefix (without extension)",
    )
    args = parser.parse_args()

    in_h5 = Path(args.in_h5).expanduser().resolve()
    out_prefix = _normalize_output_prefix(Path(args.out_prefix).expanduser().resolve())
    config_path = Path(args.config_path).expanduser().resolve()

    if not in_h5.exists():
        raise FileNotFoundError(f"Input weights file not found: {in_h5}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    model = build_model(config_path, args.image_type, args.type)
    model.load_weights(str(in_h5))

    ckpt = tf.train.Checkpoint(model=model)
    written_prefix = ckpt.write(str(out_prefix))

    index_file = Path(written_prefix + ".index")
    data_file = Path(written_prefix + ".data-00000-of-00001")

    print("Conversion complete.")
    print(f"Input .weights.h5 : {in_h5}")
    print(f"Output prefix     : {written_prefix}")
    print(f"Index file        : {index_file}")
    print(f"Data file         : {data_file}")


if __name__ == "__main__":
    main()
