import os
import json
import networkx as nx
from multiprocessing import Pool, cpu_count
from functools import partial

# Load sensitive API signatures
with open('sen.json', 'r') as f:
    sensitive_apis = set(json.load(f))

def extract_function_calls(smali_dir):
    function_calls = {}
    for root, _, files in os.walk(smali_dir):
        for file in files:
            if file.endswith('.smali'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        current_function = None
                        for line in f:
                            line = line.strip()
                            if line.startswith('.method'):
                                current_function = line.split()[-1]
                                function_calls[current_function] = set()
                            elif line.startswith('invoke-') and current_function:
                                parts = line.split(',')
                                full_call = parts[-1].strip()
                                try:
                                    class_method = full_call.split('->')[0] + '->' + full_call.split('->')[1].split('(')[0]
                                    function_calls[current_function].add(class_method)
                                except:
                                    pass
                except Exception as e:
                    print(f" Error reading {filepath}: {e}")
    return function_calls

def detect_sensitive_and_ancestors(function_calls):
    graph = nx.DiGraph()
    sensitive_nodes = set()

    for caller, callees in function_calls.items():
        for callee in callees:
            graph.add_edge(caller, callee)
            if callee in sensitive_apis:
                sensitive_nodes.add(callee)
                sensitive_nodes.add(caller)

    ancestors = set()
    for node in sensitive_nodes:
        ancestors.update(nx.ancestors(graph, node))

    return sensitive_nodes, ancestors, graph

def compute_structural_features(graph):
    features = {}

    deg_cent = nx.degree_centrality(graph)
    in_deg = dict(graph.in_degree())
    out_deg = dict(graph.out_degree())
    closeness = nx.closeness_centrality(graph)
    betweenness = nx.betweenness_centrality(graph)
    try:
        katz = nx.katz_centrality(graph, alpha=0.01, max_iter=1000)
    except:
        katz = {n: 0.0 for n in graph.nodes}
    harmonic = nx.harmonic_centrality(graph)
    pagerank = nx.pagerank(graph)
    clustering = nx.clustering(graph.to_undirected())
    square_cluster = nx.square_clustering(graph.to_undirected())

    for node in graph.nodes:
        features[node] = {
            'degree_centrality': deg_cent.get(node, 0),
            'in_degree': in_deg.get(node, 0),
            'out_degree': out_deg.get(node, 0),
            'closeness': closeness.get(node, 0),
            'betweenness': betweenness.get(node, 0),
            'katz': katz.get(node, 0),
            'harmonic': harmonic.get(node, 0),
            'pagerank': pagerank.get(node, 0),
            'clustering': clustering.get(node, 0),
            'square_clustering': square_cluster.get(node, 0)
        }

    return features
def process_apk_folder(apk_smali_path, apk_name, output_dir):
    function_calls = extract_function_calls(apk_smali_path)
    sensitive_nodes, ancestors, graph = detect_sensitive_and_ancestors(function_calls)
    features = compute_structural_features(graph)

    output = {
        'apk_name': apk_name,
        'sensitive_nodes': list(sensitive_nodes),
        'ancestors': list(ancestors),
        'structural_features': features
    }

    output_file = os.path.join(output_dir, f"{apk_name}_features.json")
    
    # Write with UTF-8 encoding and allow Unicode characters
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Processed: {apk_name} → Features saved to {output_file}")


def worker_process(args, output_dir):
    apk_path, apk_name = args
    try:
        print(f"[PID {os.getpid()}] Processing {apk_name}")
        process_apk_folder(apk_path, apk_name, output_dir)
    except Exception as e:
        print(f"Error processing {apk_name}: {e}")

def main():
    input_dir = "D:/VScode programs/Mini/smali_output_all_2021"
    output_dir = "D:/VScode programs/Mini/graph_features_all_2021"
    os.makedirs(output_dir, exist_ok=True)

    apk_folders = []
    for apk_folder in os.listdir(input_dir):
        apk_path = os.path.join(input_dir, apk_folder, "smali")
        if os.path.isdir(apk_path):
            apk_folders.append((apk_path, apk_folder))

    print(f"Using {cpu_count()} CPU cores for multiprocessing...\n")
    with Pool(processes=cpu_count()) as pool:
        pool.map(partial(worker_process, output_dir=output_dir), apk_folders)

if __name__ == "__main__":
    main()
