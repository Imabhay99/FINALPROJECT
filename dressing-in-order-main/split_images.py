import os
import shutil

base_dir = os.path.dirname(__file__)
image_dir = os.path.join(base_dir, 'datasets', 'deepfashion', 'images')
train_lst = os.path.join(base_dir, 'datasets', 'deepfashion', 'train.lst')
test_lst = os.path.join(base_dir, 'datasets', 'deepfashion', 'test.lst')
train_out = os.path.join(base_dir, 'datasets', 'train')
test_out = os.path.join(base_dir, 'datasets', 'test')
missing_log = os.path.join(base_dir, 'missing_images.txt')

# Ensure output directories exist
os.makedirs(train_out, exist_ok=True)
os.makedirs(test_out, exist_ok=True)

missing_files = []

def copy_images(lst_file, dest_folder):
    copied = 0
    with open(lst_file, 'r') as f:
        for line in f:
            filename = line.strip()
            src = os.path.join(image_dir, filename)
            dst = os.path.join(dest_folder, filename)
            os.makedirs(os.path.dirname(dst), exist_ok=True)

            if os.path.exists(src):
                shutil.copy2(src, dst)
                copied += 1
            else:
                print(f"❌ Not found: {filename}")
                missing_files.append(filename)
    return copied

print("📁 Copying training images...")
train_count = copy_images(train_lst, train_out)

print("📁 Copying testing images...")
test_count = copy_images(test_lst, test_out)

# Log missing files
if missing_files:
    with open(missing_log, 'w') as f:
        for missing in missing_files:
            f.write(missing + "\n")
    print(f"⚠️ Missing files logged in: {missing_log}")
else:
    print("✅ No missing files!")

print(f"\n✅ Done: {train_count} training images and {test_count} testing images copied.")
