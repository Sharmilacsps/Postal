import os
import requests
import csv
import time
from queue import Queue
from threading import Thread
from collections import defaultdict

# Your AndroZoo API key
API_KEY = "2af4e787c826288bf5a55a032371c33751e0e0e3eabf9a55b7702f4931afed4a".strip()

# Path to the filtered CSV file containing SHA256 hashes, year, vt_detection, and apk_size
CSV_FILE = "D:/VScode programs/Mini/sampled_apks.csv"  # Replace with your CSV file

# Directory to save the downloaded APKs
OUTPUT_DIR = "./2020"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# AndroZoo API download URL
BASE_URL = "https://androzoo.uni.lu/api/download"

# Target year for downloads
TARGET_YEAR = 2020

# Samples to download per category
SAMPLES_PER_CATEGORY = 500

# Size threshold (in MB)
SIZE_THRESHOLD_MB = 10

# Number of threads
NUM_THREADS = 5

# Queue for multithreading
download_queue = Queue()

# Dictionary to track downloaded samples per category
downloaded_counts = defaultdict(int)


def parse_vt_detection(value):
    """Parse and validate vt_detection value."""
    try:
        return int(float(value))  # Convert from float if necessary
    except ValueError:
        return None  # Return None if the value is invalid


def download_apk(sha256, output_dir, category):
    """Download an APK using the AndroZoo API."""
    try:
        output_path = os.path.join(output_dir, f"{sha256}.apk")

        # Skip if APK is already downloaded
        if os.path.exists(output_path):
            print(f"Skipped (already exists): {sha256} ({category})")
            return

        # Construct the request URL with the API key and SHA256 hash
        url = f"{BASE_URL}?apikey={API_KEY}&sha256={sha256}"

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(output_path, "wb") as apk_file:
                for chunk in response.iter_content(chunk_size=1024):
                    apk_file.write(chunk)
            print(f"Downloaded: {sha256} ({category})")
        else:
            print(f"Failed to download {sha256}: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error downloading {sha256}: {e}")


def worker():
    """Worker thread to process the download queue."""
    while not download_queue.empty():
        sha256, category, output_dir = download_queue.get()
        try:
            download_apk(sha256, output_dir, category)
        except Exception as e:
            print(f"Worker error processing {sha256}: {e}")
        finally:
            download_queue.task_done()


def main():
    """Main function to enqueue downloads and manage threads."""
    try:
        with open(CSV_FILE, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)

            # Process rows and enqueue APKs for downloading
            for row in reader:
                try:
                    year = int(row['year'])  # Ensure 'year' column exists in the CSV
                    sha256 = row['sha256']

                    # Parse and validate vt_detection and apk_size
                    vt_detection = parse_vt_detection(row['vt_detection'])
                    apk_size_mb = float(row['apk_size']) / (1024 * 1024)  # Convert size from bytes to MB

                    if vt_detection is None or apk_size_mb > SIZE_THRESHOLD_MB or year != TARGET_YEAR:
                        continue  # Skip invalid rows, large APKs, or non-target years

                    # Categorize APK based on vt_detection threshold
                    category = 'benign' if vt_detection == 0 else 'malicious'

                    # Check category limits
                    if downloaded_counts[category] < SAMPLES_PER_CATEGORY:
                        category_dir = os.path.join(OUTPUT_DIR, str(TARGET_YEAR), category)
                        os.makedirs(category_dir, exist_ok=True)

                        # Enqueue the download if not already downloaded
                        output_path = os.path.join(category_dir, f"{sha256}.apk")
                        if not os.path.exists(output_path):
                            download_queue.put((sha256, category, category_dir))
                            downloaded_counts[category] += 1
                except KeyError as e:
                    print(f"Missing column in CSV: {e}")
                except Exception as e:
                    print(f"Error processing row: {e}")

        # Create and start threads
        threads = []
        for _ in range(NUM_THREADS):
            thread = Thread(target=worker)
            thread.start()
            threads.append(thread)

        # Wait for all threads to finish
        download_queue.join()
        for thread in threads:
            thread.join()

        print("All downloads for 2020 completed.")

    except FileNotFoundError as e:
        print(f"CSV file not found: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    print(f"Using API Key: {API_KEY}")
    main()
