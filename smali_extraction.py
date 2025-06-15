import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

DEX_DIR = r"D:/VScode programs/Mini/extracted_dex_code_all_2020/"  
SMALI_OUTPUT_DIR = r"D:/VScode programs/Mini/smali_output_all_2020/"  
BAKSMLI_JAR = r"D:/VScode programs/Mini/baksmali-2.5.2.jar"  
NUM_THREADS = 8 

def extract_smali(dex_file):
    """
    Extracts Smali code from a single DEX file using Baksmali.
    """
    try:
        # Get APK name from DEX filename (remove ".dex")
        apk_name = os.path.basename(dex_file).replace(".dex", "")

       
        output_path = os.path.join(SMALI_OUTPUT_DIR, apk_name, "smali")

        
        os.makedirs(output_path, exist_ok=True)

        command = f'java -jar "{BAKSMLI_JAR}" d "{dex_file}" -o "{output_path}"'
        subprocess.run(command, shell=True, check=True)

        print(f"Successfully processed: {apk_name}")
        return apk_name

    except Exception as e:
        print(f"Error extracting Smali for {dex_file}: {e}")
        return None

def process_all_dex():
    """
    Process all standalone DEX files in the specified directory using multithreading.
    """
 
    dex_files = [os.path.join(DEX_DIR, f) for f in os.listdir(DEX_DIR) if f.endswith(".dex")]

    if not dex_files:
        print("No DEX files found. Check the directory path and file naming.")
        return

    print(f"Found {len(dex_files)} DEX files. Starting extraction...")

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        results = list(executor.map(extract_smali, dex_files))

    print(f"Extraction complete! Successfully processed {len([r for r in results if r])} APKs.")

if __name__ == "__main__":
    process_all_dex()
