import os
import glob
import struct
import argparse

def verify_bin_idx(bin_path, idx_path):
    if not os.path.exists(bin_path):
        return False, f"{bin_path} missing"
    if not os.path.exists(idx_path):
        return False, f"{idx_path} missing"

    # Check sizes
    bin_size = os.path.getsize(bin_path)
    idx_size = os.path.getsize(idx_path)

    if bin_size == 0: return False, f"{bin_path} empty"
    if idx_size == 0: return False, f"{idx_path} empty"

    try:
        with open(bin_path, 'rb') as f:
            f.read(100)
        with open(idx_path, 'rb') as f:
            f.read(100)
    except Exception as e:
        return False, f"Read error: {e}"

    return True, "OK"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    print(f"Verifying data in {args.data_dir}...")
    if not os.path.exists(args.data_dir):
        print(f"Data directory '{args.data_dir}' not found.")
        return

    # Recursive search for .bin files
    bin_files = []
    for root, dirs, files in os.walk(args.data_dir):
        for f in files:
            if f.endswith(".bin"):
                bin_files.append(os.path.join(root, f))

    if not bin_files:
        print("No .bin files found.")
        return

    all_ok = True
    for bin_f in bin_files:
        idx_f = bin_f.replace(".bin", ".idx")
        if os.path.exists(idx_f):
            ok, msg = verify_bin_idx(bin_f, idx_f)
            status = "✅" if ok else "❌"
            print(f"{status} {bin_f} | {msg}")
            if not ok: all_ok = False
        else:
            # Just verify bin
            try:
                with open(bin_f, 'rb') as f: f.read(10)
                print(f"⚠️ {bin_f} (No .idx found, but readable)")
            except:
                print(f"❌ {bin_f} (Read error)")
                all_ok = False

    if all_ok:
        print("\nData verification passed.")
    else:
        print("\nData verification failed for some files.")

if __name__ == "__main__":
    main()
