import os
import re
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def clean_smali_line(line):
    # 1. Eliminating Variable Markings
    line = re.sub(r'\b[vp][0-9]+\b', '', line)
    
    # 2. Eliminating Special Characters
    line = re.sub(r'[^\w\s\->/;]', ' ', line)
    
    # 3. Eliminating Single-Character Strings
    # Split the line into tokens and remove any single-character tokens that likely do not add semantic value.
    tokens = [token for token in line.split() if len(token) > 1]
    line = ' '.join(tokens)
    
    # 4. Eliminating Comments and Alerts
    # Remove comments starting with # since they don't affect code behavior and add noise.
    line = re.sub(r'#.*', '', line)
    
    # 4. (continued) Eliminating const-string lines with multiple strings
    # Remove lines containing const-string instructions with multiple string parts to reduce clutter.
    line = re.sub(r'const-string.*",.*', '', line)
    
    # 5. Eliminating Redundant Whitespace and Trimming
    # Replace multiple spaces with a single space and strip leading/trailing whitespace.
    return re.sub(r'\s+', ' ', line).strip()

def process_smali_file(file_path):
    cleaned_lines = []
    seen_lines = set()
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            cleaned = clean_smali_line(line)
            if cleaned and cleaned not in seen_lines:
                cleaned_lines.append(cleaned)
                seen_lines.add(cleaned)
    return cleaned_lines

def process_apk(apk_folder_path, apk_name, output_folder):
    apk_cleaned_lines = []
    for root, _, files in os.walk(apk_folder_path):
        for file in files:
            if file.endswith('.smali'):
                file_path = os.path.join(root, file)
                apk_cleaned_lines.extend(process_smali_file(file_path))

    out_path = os.path.join(output_folder, f"{apk_name}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(apk_cleaned_lines, f, indent=2)

    return apk_name  # just returning the name for logging

def preprocess_all_apks(smali_root, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    apk_folders = [folder for folder in os.listdir(smali_root) if os.path.isdir(os.path.join(smali_root, folder))]

    with ProcessPoolExecutor() as executor:
        futures = []
        for apk_folder in apk_folders:
            apk_path = os.path.join(smali_root, apk_folder)
            futures.append(executor.submit(process_apk, apk_path, apk_folder, output_folder))

        for future in tqdm(futures, total=len(futures), desc="Processing APKs"):
            future.result()  # You can log result if needed

if __name__ == '__main__':
    smali_root = r"D:\VScode programs\Mini\smali_output_all_2021"
    output_folder = r"D:\VScode programs\Mini\cleaned_smali_json_2021"
    preprocess_all_apks(smali_root, output_folder)
    print(f"\n Cleaning complete. JSONs saved to: {output_folder}")
