"""
Convert model weights to legacy TensorFlow checkpoint files for old TF2.0 project.

Output format:
  <out_prefix>.index
  <out_prefix>.data-00000-of-00001

Run this in the OLD Windows project environment (TensorFlow 2.0) so layer naming
and checkpoint compatibility match the old codebase.

Examples:
  python convert_to_legacy.py ^
    --in_model checkpoints2\\rgb_Dtt_Dxx\\EstimationModel_epoch_100.weights.h5 ^
    --out_prefix checkpoints\\rgb_Dtt_Dxx\\EstimationModel_epoch_100

  python convert_to_legacy.py ^
    --in_model checkpoints\\rgb_Dtt_Dxx\\EstimationModel_epoch_100 ^
    --out_prefix checkpoints\\rgb_Dtt_Dxx\\EstimationModel_epoch_100
"""

import argparse
import os
import re
import sys

import yaml
from pathlib import Path

sys.path.insert(0, "..")

import libs.yaml_utils as yaml_utils
from libs.EstimatiorModel import TemplateEstimatior


def _normalize_input_candidates(path):
    p = os.path.normpath(path)
    if p.endswith(".index"):
        p = p[:-len(".index")]
    elif p.endswith(".data-00000-of-00001"):
        p = p[:-len(".data-00000-of-00001")]

    # If caller provided explicit .h5 path, try it first.
    if p.endswith(".weights.h5") or p.endswith(".h5"):
        base = p[:-len(".weights.h5")] if p.endswith(".weights.h5") else p[:-len(".h5")]
        return [p, base, base + ".weights.h5", base + ".h5"]

    return [p, p + ".weights.h5", p + ".h5"]


def _resolve_path_candidates(raw_path):
    """
    Resolve common user-provided relative paths when run from different dirs.
    """
    candidates = []
    p = Path(raw_path)
    candidates.append(p)

    # If user included "one_class_classification/..." while already in that dir.
    if "one_class_classification" in p.parts:
        parts = list(p.parts)
        idx = parts.index("one_class_classification")
        candidates.append(Path(*parts[idx + 1 :]))

    script_dir = Path(__file__).resolve().parent
    candidates.append(script_dir / p)
    candidates.append(script_dir.parent / p)

    # Deduplicate while preserving order
    seen = set()
    out = []
    for c in candidates:
        c = Path(c)
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def _load_model_weights(model, in_model):
    raw_candidates = _normalize_input_candidates(in_model)
    candidates = []
    for raw in raw_candidates:
        for resolved in _resolve_path_candidates(raw):
            candidates.append(os.path.normpath(str(resolved)))
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

            status = model.load_weights(path)
            if hasattr(status, "expect_partial"):
                status.expect_partial()
            return path
        except Exception as exc:
            one_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
            errors.append("%s -> %s: %s" % (path, type(exc).__name__, one_line))

    raise RuntimeError(
        "Could not load source model.\nTried:\n  - %s\nErrors:\n  - %s"
        % ("\n  - ".join(candidates), "\n  - ".join(errors) if errors else "none")
    )


def _default_out_prefix(in_model, checkpoint_dir):
    in_model_norm = os.path.normpath(in_model)
    m = re.search(r"EstimationModel_epoch_(\d+)", in_model_norm)
    epoch = m.group(1) if m else "converted"
    return os.path.join(checkpoint_dir, "EstimationModel_epoch_%s" % epoch)


def main():
    parser = argparse.ArgumentParser(description="Convert model to legacy TF checkpoint format")
    parser.add_argument("--config_path", default="./configuration.yml", type=str, help="Path to config YAML")
    parser.add_argument("--image_type", default="rgb", choices=["rgb", "gray"], help="Model image type")
    parser.add_argument("--type", default="Dtt_Dxx", choices=["Dtt_Dxx", "Dtt_Dt_Dxx_Dx"], help="Model type")
    parser.add_argument("--in_model", required=True, type=str, help="Source model path (.weights.h5/.h5 or checkpoint prefix)")
    parser.add_argument("--out_prefix", default="", type=str, help="Output checkpoint prefix (no extension)")
    parser.add_argument("--cpu_only", action="store_true", help="Disable GPU for this run")
    args = parser.parse_args()

    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if not os.path.exists(args.config_path):
        raise FileNotFoundError("Config not found: %s" % args.config_path)

    config = yaml_utils.Config(yaml.load(open(args.config_path), Loader=yaml.FullLoader))

    # Build model exactly as old pipeline does.
    runtime_args = argparse.Namespace(
        image_type=args.image_type,
        type=args.type,
        checkpoint_dir="%s_%s" % (args.image_type, args.type),
        dir=args.type,
    )
    estimator = TemplateEstimatior(config, runtime_args, type=args.type)
    model = estimator.EstimationModel

    loaded_from = _load_model_weights(model, args.in_model)

    out_prefix = args.out_prefix.strip()
    if not out_prefix:
        out_prefix = _default_out_prefix(args.in_model, estimator.checkpoint_dir)
    out_prefix = os.path.normpath(out_prefix)

    out_dir = os.path.dirname(out_prefix)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # IMPORTANT: write legacy TF checkpoint pair (.index + .data...) using Checkpoint API
    # Keras 3 enforces .weights.h5 filenames, so use tf.train.Checkpoint.
    import tensorflow as tf
    ckpt = tf.train.Checkpoint(model=model)
    out_prefix = ckpt.write(out_prefix)

    index_file = out_prefix + ".index"
    data_file = out_prefix + ".data-00000-of-00001"

    if not (os.path.exists(index_file) and os.path.exists(data_file)):
        raise RuntimeError("Conversion did not produce expected files:\n%s\n%s" % (index_file, data_file))

    print("\nConversion complete.")
    print("Loaded from : %s" % loaded_from)
    print("Output      : %s" % out_prefix)
    print("Created     : %s" % index_file)
    print("Created     : %s" % data_file)


if __name__ == "__main__":
    main()
