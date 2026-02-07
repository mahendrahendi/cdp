"""
Helper script to duplicate template 0 for all photos
Run this after adding new photos to data/original/rgb/
"""

import os
import shutil

def duplicate_templates():
    binary_dir = "data/binary"
    rgb_dir = "data/original/rgb"

    # Get master template
    master_template = os.path.join(binary_dir, "0000.png")

    if not os.path.exists(master_template):
        print(f"ERROR: Master template not found: {master_template}")
        return

    # Get all photo files
    photo_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])

    print(f"Found {len(photo_files)} photos in {rgb_dir}")
    print(f"Master template: {master_template}")
    print("\nDuplicating template for each photo...")

    for photo_file in photo_files:
        template_file = os.path.join(binary_dir, photo_file)

        if os.path.exists(template_file):
            print(f"  ✓ {photo_file} - already exists")
        else:
            shutil.copy(master_template, template_file)
            print(f"  + {photo_file} - created")

    # Verify counts match
    template_files = [f for f in os.listdir(binary_dir) if f.endswith('.png')]

    print(f"\nSummary:")
    print(f"  Photos: {len(photo_files)}")
    print(f"  Templates: {len(template_files)}")
    print(f"  Match: {'✓ Yes' if len(photo_files) == len(template_files) else '✗ No'}")

if __name__ == "__main__":
    duplicate_templates()
