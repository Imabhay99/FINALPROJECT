import os
import gdown
import zipfile
import shutil
from tqdm import tqdm

def safe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def download_from_gdrive(dst_root, fn, gdrive_id, iszip=True):
    safe_mkdir(dst_root)
    
    zip_path = f"{fn}.zip" if iszip else fn
    output_path = os.path.join(dst_root, fn)

    # Download only if it doesn't exist
    if not os.path.exists(output_path):
        print(f"📥 Downloading {fn}...")
        gdown.download(id=gdrive_id, output=zip_path, quiet=False)

        if iszip:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dst_root)
            os.remove(zip_path)

            # Move folder if extracted outside
            if not os.path.exists(output_path) and os.path.exists(fn):
                shutil.move(fn, dst_root)

    print(f"✅ Downloaded and ready: {fn}")

# 1. Download all data
safe_mkdir("data")

download_from_gdrive("data", "testM_lip", "1toeQwAe57LNPTy9EWGG0u1XfTI7qv6b1")
download_from_gdrive("data", "images", "1U2PljA7NE57jcSSzPs21ZurdIPXdYZtN")
download_from_gdrive("data", "fasion-pairs-test.csv", "12fZKGf0kIu5OX3mjC-C3tptxrD8sxm7x", iszip=False)
download_from_gdrive("data", "fasion-annotation-test.csv", "1MxkVFFtNsWFshQp_TA7qwIGEUEUIpYdS", iszip=False)
download_from_gdrive("data", "standard_test_anns.txt", "19nJSHrQuoJZ-6cSl3WEYlhQv6ZsAYG-X", iszip=False)

# 2. Filter test images based on mask availability
print("🔍 Filtering test images...")
test_mask_dir = "data/testM_lip"
image_dir = "data/images"
filtered_dir = "data/test"

safe_mkdir(filtered_dir)

# Get all test mask filenames without `.png`
target_fns = [fn[:-4] for fn in os.listdir(test_mask_dir) if fn.endswith('.png')]

for fn in tqdm(os.listdir(image_dir)):
    elements = fn.split("-")
    if len(elements) < 4:
        continue  # Skip malformed names

    elements[2] = elements[2].replace("_", "")
    last_elements = elements[-1].split("_")
    if len(last_elements) < 3:
        continue

    elements[-1] = f"{last_elements[0]}_{last_elements[1]}{last_elements[2]}"
    new_fn = "fashion" + "".join(elements)

    if new_fn[:-4] in target_fns:
        src = os.path.join(image_dir, fn)
        dst = os.path.join(filtered_dir, new_fn)
        shutil.move(src, dst)

print("✅ Image filtering completed and saved in `data/test`.")
