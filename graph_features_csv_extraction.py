import os
import json
import pandas as pd

def load_features_from_json(json_dir):
    data = []

    for file in os.listdir(json_dir):
        if file.endswith(".json"):
            file_path = os.path.join(json_dir, file)
            with open(file_path, 'r') as f:
                content = json.load(f)

                apk_name = content.get('apk_name', 'unknown_apk')
                label = 0 if apk_name.startswith('benign_') else 1  # 0: benign, 1: malicious

                features_dict = content.get('structural_features', {})
                for method, features in features_dict.items():
                    row = {
                        'apk_name': apk_name,
                        'method_name': method,
                        'label': label
                    }
                    row.update(features)  # Add all structural features
                    data.append(row)

    df = pd.DataFrame(data)
    return df

def main():
    json_dir = "D:/VScode programs/Mini/graph_features_all"  # Replace with your actual path
    output_csv = "D:/VScode programs/Mini/graph_features_dataset.csv"

    df = load_features_from_json(json_dir)

    print(f"✅ Loaded {len(df)} rows with {len(df.columns)} columns.")
    print(df.head())

    df.to_csv(output_csv, index=False)
    print(f"📁 CSV saved to: {output_csv}")

if __name__ == "__main__":
    main()
