import os
import gzip
import shutil

DATA_DIR = "data"
MAX_COMPRESSIONS = 3

def compress_pickle_file(file_path):
    temp_file = file_path + ".gz"

    for iteration in range(1, MAX_COMPRESSIONS + 1):
        print(f"\n🔄 Iteration {iteration}: Compressing {file_path}")
        original_size = os.path.getsize(file_path)

        with open(file_path, "rb") as f_in:
            with gzip.open(temp_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        compressed_size = os.path.getsize(temp_file)
        saved = (1 - (compressed_size / original_size)) * 100
        print(f"✅ Done: {temp_file} ({compressed_size / (1024**2):.2f} MB, saved {saved:.1f}%)")

        os.remove(file_path)
        shutil.move(temp_file, file_path)

        if iteration > 1 and abs(prev_size - compressed_size) < 1024:
            print("⚠️ Compression gain negligible, stopping early.")
            break

        prev_size = compressed_size

    print(f"🎉 Finished compressing {file_path} (max {iteration} iterations).")

def main():
    if not os.path.exists(DATA_DIR):
        print(f"❌ Folder '{DATA_DIR}' not found.")
        return

    pkl_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pkl")]
    if not pkl_files:
        print("⚠️ No .pkl files found.")
        return

    for filename in pkl_files:
        compress_pickle_file(os.path.join(DATA_DIR, filename))

if __name__ == "__main__":
    main()
