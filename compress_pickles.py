import os
import gzip
import shutil

DATA_DIR = "data"  # Folder containing your .pkl files
MAX_COMPRESSIONS = 3  # Maximum number of compression iterations per file

def compress_pickle_file(file_path):
    """Compress a .pkl file up to MAX_COMPRESSIONS times using gzip."""
    
    temp_file = file_path + ".gz"

    for iteration in range(1, MAX_COMPRESSIONS + 1):
        print(f"\n🔄 Iteration {iteration}: Compressing {file_path}")
        original_size = os.path.getsize(file_path)

        # Compress with gzip
        with open(file_path, "rb") as f_in:
            with gzip.open(temp_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        compressed_size = os.path.getsize(temp_file)
        saved = (1 - (compressed_size / original_size)) * 100

        print(f"✅ Done: {temp_file}")
        print(f"📦 Original: {original_size / (1024**2):.2f} MB")
        print(f"💨 Compressed: {compressed_size / (1024**2):.2f} MB")
        print(f"💾 Space saved: {saved:.1f}%")

        # Move compressed file to replace original for next iteration
        os.remove(file_path)
        shutil.move(temp_file, file_path)

        # Stop early if compression no longer reduces size significantly
        if iteration > 1 and abs(prev_size - compressed_size) < 1024:  # less than 1 KB difference
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
        print("⚠️ No .pkl files found to compress.")
        return

    print(f"Found {len(pkl_files)} .pkl files in '{DATA_DIR}': {pkl_files}")

    for filename in pkl_files:
        compress_pickle_file(os.path.join(DATA_DIR, filename))

    print("\n🎉 All files compressed successfully!")
    print("🗑️ Original .pkl files replaced with compressed versions.")

if __name__ == "__main__":
    main()
