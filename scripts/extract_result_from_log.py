import os
import shutil


def extract_result_from_log(dir, outdir):
    """Extract result from a experiment set"""
    os.makedirs(outdir, exist_ok=True)
    dirs = [d for d in sorted(os.listdir(dir)) if os.path.isdir(os.path.join(dir, d))]

    for d in dirs:
        di = os.path.join(dir, d)
        test_dir = os.path.join(di, "images_test")
        test_iters = [test for test in sorted(os.listdir(test_dir)) if os.path.isdir(os.path.join(test_dir, test))]
        dir_last = os.path.join(test_dir, test_iters[-1])
        img = [img for img in sorted(os.listdir(dir_last)) if img.startswith("img")]
        mid = img[len(img) // 2]
        mid_path = os.path.join(dir_last, mid)
        to_path = os.path.join(outdir, f"{int(d):06}.png")

        print(f"{mid_path} -> {to_path}")
        shutil.copy(mid_path, to_path)

    print(f"Copy of {dir} to {outdir} success")


if __name__ == '__main__':
    input_dir = os.path.expanduser("../logs/compare")
    output_dir = os.path.expanduser("../logs/compare_out")
    extract_result_from_log(input_dir, output_dir)