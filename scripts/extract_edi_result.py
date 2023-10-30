import os
import shutil

import configargparse


def extract_result_from_log(dir, outdir):
    """Extract result from a experiment set"""
    assert os.path.exists(dir)
    os.makedirs(outdir, exist_ok=True)
    files = [f for f in sorted(os.listdir(dir)) if f.lower().endswith(("51.png", ".jpg", ".jpeg"))]

    for i, f in enumerate(files):
        to_path = os.path.join(outdir, f"{int(i):06}.png")
        from_path = os.path.join(dir, f)
        shutil.copy(from_path, to_path)
        print(f"{from_path} -> {to_path}")

    print(f"Copy of {dir} to {outdir} success")


if __name__ == '__main__':
    parser = configargparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/Users/pianwan/Downloads/livingroom_gray/im")
    parser.add_argument("--output_dir", type=str, default="/Users/pianwan/Downloads/livingroom_gray/out")

    args = parser.parse_args()

    input_dir = os.path.expanduser(args.input_dir)
    output_dir = os.path.expanduser(args.output_dir)
    extract_result_from_log(input_dir, output_dir)
