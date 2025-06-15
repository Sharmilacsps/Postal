from androguard.core.bytecodes.apk import APK
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Base Paths
apk_base_directory = r"D:\VScode programs\Mini\2020"
output_directory = r"D:\VScode programs\Mini\extracted_dex_code_all_2020"
num_threads = 8  # Adjust as needed

def extract_valid_dex(apk_path, label):
    """
    Extracts the DEX file from a single APK and saves it with a label prefix.
    :param apk_path: Path to the APK file.
    :param label: 'benign' or 'malicious' label prefix.
    """
    apk_name = os.path.basename(apk_path).replace(".apk", "")
    apk_name = f"{label}_{apk_name}"  # Add prefix
    dex_output_path = os.path.join(output_directory, f"{apk_name}.dex")

    try:
        app = APK(apk_path)
        if not app.is_valid_APK():
            print(f" Invalid APK: {apk_name}")
            return None

        dex_data = app.get_dex()
        if dex_data:
            with open(dex_output_path, "wb") as f:
                f.write(dex_data)
            print(f"Extracted: {dex_output_path}")
            return dex_output_path
        else:
            print(f" No valid DEX in {apk_name}")
            return None
    except Exception as e:
        print(f" Error processing {apk_name}: {e}")
        return None

def process_all_apks(base_dir, num_threads):
    """
    Processes all APKs in benign and malicious subfolders using multithreading.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    subdirs = {'benign': 'benign', 'malicious': 'malicious'}
    all_tasks = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for label, subfolder in subdirs.items():
            folder = os.path.join(base_dir, '2020', subfolder)
            apk_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.apk')]
            print(f"[{label.upper()}] Found {len(apk_files)} APKs in {folder}")

            for apk in apk_files:
                future = executor.submit(extract_valid_dex, apk, label)
                all_tasks.append((apk, future))

        for apk_path, future in all_tasks:
            try:
                result = future.result()
                if result:
                    print(f"Processed: {os.path.basename(apk_path)}")
            except Exception as e:
                print(f" Failed: {os.path.basename(apk_path)} | Error: {e}")

if __name__ == "__main__":
    process_all_apks(apk_base_directory, num_threads)
