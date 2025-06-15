import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import subprocess
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
from androguard.core.bytecodes.apk import APK
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for non-interactive backend
import matplotlib
matplotlib.use('Agg')


class APKClassifier:
    def __init__(self, model_path="models/ensemble_model1.joblib", 
                 scaler_path="models/scaler1.joblib",
                 feature_names_path="models/feature_names1.joblib",
                 sensitive_apis_path="sen.json",
                 baksmali_jar="baksmali-2.5.2.jar"):
        """
        Initialize the APK classifier with trained model and required files.
        
        Args:
            model_path: Path to the trained ensemble model
            scaler_path: Path to the feature scaler
            feature_names_path: Path to the feature names file
            sensitive_apis_path: Path to sensitive APIs JSON file
            baksmali_jar: Path to baksmali JAR file
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.feature_names_path = feature_names_path
        self.sensitive_apis_path = sensitive_apis_path
        self.baksmali_jar = baksmali_jar
        
        # Create output directory
        self.output_dir = "classification_results/res"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load model components
        self._load_model_components()
        
        # Load sensitive APIs
        self._load_sensitive_apis()
        
        # Initialize SHAP explainer (will be created when first used)
        self.explainer = None
        
        # Define permission-based features that need to be extracted
        self.permission_features = [
            'uses_SEND_SMS', 'uses_READ_SMS', 'uses_RECEIVE_SMS', 'uses_WRITE_SMS',
            'uses_READ_PHONE_STATE', 'uses_getDeviceId', 'uses_getSubscriberId',
            'uses_RECEIVE_BOOT_COMPLETED', 'uses_USE_FINGERPRINT', 'uses_READ_CONTACTS',
            'uses_CALL_PHONE', 'uses_ACCESS_FINE_LOCATION', 'uses_RECORD_AUDIO',
            'uses_CAMERA', 'uses_READ_EXTERNAL_STORAGE', 'uses_ACCESS_WIFI_STATE',
            'uses_INTERNET', 'uses_QUERY_ALL_PACKAGES'
        ]

    def _create_visualizations(self, apk_name, prediction, probability, shap_explanation, features):
        """Create visualization plots."""
        print("Creating visualizations...")

        visualization_paths = {}

        try:
            # 1. Feature importance bar plot
            if 'top_10_features' in shap_explanation:
                plt.figure(figsize=(14, 10))
                top_features = shap_explanation['top_10_features']

                feature_names = [f['feature'] for f in top_features]
                importance_values = [f['importance'] for f in top_features]

                colors = ['red' if f['shap_value'] > 0 else 'blue' for f in top_features]

                plt.barh(range(len(feature_names)), importance_values, color=colors, alpha=0.7)
                plt.yticks(range(len(feature_names)), feature_names)
                plt.xlabel('SHAP Value Magnitude')
                plt.title(f'Top 10 Important Features - {apk_name}\n'
                        f'Prediction: {"Malicious" if prediction == 1 else "Benign"} '
                        )
                plt.tight_layout()

                importance_plot_path = os.path.join(self.output_dir, f"{apk_name}_feature_importance.png")
                plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                visualization_paths['feature_importance'] = importance_plot_path

            # 2. Permission-based features visualization
            if any(feature in features for feature in self.permission_features):
                plt.figure(figsize=(12, 8))
                permission_data = []
                permission_labels = []
                
                for perm in self.permission_features:
                    if perm in features and features[perm] > 0:
                        permission_data.append(features[perm])
                        permission_labels.append(perm.replace('uses_', ''))
                
                if permission_data:
                    plt.barh(range(len(permission_labels)), permission_data, color='orange', alpha=0.7)
                    plt.yticks(range(len(permission_labels)), permission_labels)
                    plt.xlabel('Permission Usage')
                    plt.title(f'Permission-based Features - {apk_name}')
                    plt.tight_layout()

                    perm_plot_path = os.path.join(self.output_dir, f"{apk_name}_permissions.png")
                    plt.savefig(perm_plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    visualization_paths['permissions'] = perm_plot_path

            # 3. SHAP values distribution
            if 'shap_values' in shap_explanation:
                plt.figure(figsize=(10, 6))
                shap_vals = shap_explanation['shap_values']

                plt.hist(shap_vals, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                plt.xlabel('SHAP Value')
                plt.ylabel('Frequency')
                plt.title(f'Distribution of SHAP Values - {apk_name}')
                plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
                plt.tight_layout()

                dist_plot_path = os.path.join(self.output_dir, f"{apk_name}_shap_distribution.png")
                plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                visualization_paths['shap_distribution'] = dist_plot_path

            # 4. Prediction summary bar plot
            plt.figure(figsize=(8, 5))

            categories = ['Benign', 'Malicious']
            probabilities = [1 - probability, probability]
            colors = ['green', 'red']

            bars = plt.bar(categories, probabilities, color=colors, alpha=0.7)
            plt.ylabel('Probability')
            plt.title(f'Prediction Summary - {apk_name}')
            plt.ylim(0, 1)

            for bar, prob in zip(bars, probabilities):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{prob:.2%}', ha='center', va='bottom', fontweight='bold')

            plt.tight_layout()

            summary_plot_path = os.path.join(self.output_dir, f"{apk_name}_prediction_summary.png")
            plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualization_paths['prediction_summary'] = summary_plot_path

            print(" Visualizations created successfully")

        except Exception as e:
            print(f"Error creating visualizations: {e}")

        return visualization_paths

    def _load_model_components(self):
        """Load the trained model, scaler, and feature names."""
        try:
            print("Loading trained model components...")
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.feature_names = joblib.load(self.feature_names_path)
            print(f"Model components loaded successfully")
            print(f"Expected features: {len(self.feature_names)}")
        except Exception as e:
            print(f" Error loading model components: {e}")
            raise
    
    def _load_sensitive_apis(self):
        """Load sensitive API signatures."""
        try:
            if os.path.exists(self.sensitive_apis_path):
                with open(self.sensitive_apis_path, 'r') as f:
                    apis = json.load(f)
                    # Clean formatting to only keep class->method names
                    self.sensitive_apis = set()
                    for api in apis:
                        if '->' in api:
                            clean_api = api.split('(')[0]  # Remove parameters/return
                            self.sensitive_apis.add(clean_api.strip())
                print(f"Loaded {len(self.sensitive_apis)} sensitive APIs")
            else:
                print("Sensitive APIs file not found, using default set")
                self.sensitive_apis = {
                    'Landroid/telephony/TelephonyManager;->getDeviceId',
                    'Landroid/location/LocationManager;->getLastKnownLocation',
                    'Landroid/telephony/SmsManager;->sendTextMessage',
                    'Ljava/lang/Runtime;->exec',
                    'Landroid/content/pm/PackageManager;->getInstalledPackages'
                }
        except Exception as e:
            print(f" Error loading sensitive APIs: {e}")
            self.sensitive_apis = set()

    
    def classify_apk(self, apk_path):
        """
        Main method to classify an APK file.
        
        Args:
            apk_path: Path to the APK file
            
        Returns:
            Dictionary containing classification results and explanations
        """
        if not os.path.exists(apk_path):
            return {"error": f"APK file not found: {apk_path}"}
        
        print(f"Starting classification of: {os.path.basename(apk_path)}")
        
        try:
            # Step 1: Extract APK information
            apk_name = os.path.basename(apk_path).replace(".apk", "")
            temp_dir = os.path.join(self.output_dir, apk_name)
            os.makedirs(temp_dir, exist_ok=True)
            
            # Step 2: Extract features from APK
            features = self._extract_all_features(apk_path, temp_dir, apk_name)
            if features is None:
                return {"error": "Failed to extract features from APK"}
            
            # Step 3: Make prediction
            prediction, probability = self._predict(features, apk_name)
            
            # Step 4: Generate SHAP explanations
            shap_explanation = self._generate_shap_explanation(features, apk_name)
            
            # Step 5: Create visualizations
            visualization_paths = self._create_visualizations(
                apk_name, prediction, probability, shap_explanation, features
            )
            
            # Prepare final result
            result = {
                "apk_name": apk_name,
                "prediction": "Malicious" if prediction == 1 else "Benign",
                "confidence": float(probability),
                "features_extracted": len(features),
                "shap_explanation": shap_explanation,
                "visualization_paths": visualization_paths,
                "temp_directory": temp_dir,
                "feature_summary": self._get_feature_summary(features)
            }
            
            # Print summary
            self._print_classification_summary(result)
            
            return result
            
        except Exception as e:
            print(f" Error during classification: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Classification failed: {str(e)}"}
    
    def _extract_all_features(self, apk_path, temp_dir, apk_name):
        """Extract all features needed for classification."""
        try:
            print("Extracting features from APK...")
            
            # Step 1: Extract DEX file
            dex_path = self._extract_dex(apk_path, temp_dir)
            if not dex_path:
                print("Failed to extract DEX file")
                return None
            
            # Step 2: Extract smali code
            smali_dir = self._extract_smali(dex_path, temp_dir)
            if not smali_dir:
                print("Failed to extract smali code")
                return None
            
            # Step 3: Extract structural features
            structural_features = self._extract_structural_features(smali_dir)
            
            # Step 4: Extract semantic features (Word2Vec embeddings)
            semantic_features = self._extract_semantic_features(smali_dir)
            
            # Step 5: Extract permission-based features
            permission_features = self._extract_permission_features(apk_path, smali_dir)
            
            # Step 6: Combine all features
            all_features = {}
            all_features.update(structural_features)
            all_features.update(semantic_features)
            all_features.update(permission_features)
            
            print(f"Extracted {len(all_features)} features")
            return all_features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def _extract_permission_features(self, apk_path, smali_dir):
        """Extract permission-based features from APK and smali code."""
        print(" Extracting permission-based features...")
        
        permission_features = {}
        
        try:
            # Initialize all permission features to 0
            for perm in self.permission_features:
                permission_features[perm] = 0
            permission_features['num_sensitive_flags'] = 0
            
            # Extract from APK manifest
            apk = APK(apk_path)
            permissions = apk.get_permissions()
            
            # Map permissions to features
            permission_mapping = {
                'android.permission.SEND_SMS': 'uses_SEND_SMS',
                'android.permission.READ_SMS': 'uses_READ_SMS',
                'android.permission.RECEIVE_SMS': 'uses_RECEIVE_SMS',
                'android.permission.WRITE_SMS': 'uses_WRITE_SMS',
                'android.permission.READ_PHONE_STATE': 'uses_READ_PHONE_STATE',
                'android.permission.RECEIVE_BOOT_COMPLETED': 'uses_RECEIVE_BOOT_COMPLETED',
                'android.permission.USE_FINGERPRINT': 'uses_USE_FINGERPRINT',
                'android.permission.READ_CONTACTS': 'uses_READ_CONTACTS',
                'android.permission.CALL_PHONE': 'uses_CALL_PHONE',
                'android.permission.ACCESS_FINE_LOCATION': 'uses_ACCESS_FINE_LOCATION',
                'android.permission.RECORD_AUDIO': 'uses_RECORD_AUDIO',
                'android.permission.CAMERA': 'uses_CAMERA',
                'android.permission.READ_EXTERNAL_STORAGE': 'uses_READ_EXTERNAL_STORAGE',
                'android.permission.ACCESS_WIFI_STATE': 'uses_ACCESS_WIFI_STATE',
                'android.permission.INTERNET': 'uses_INTERNET',
                'android.permission.QUERY_ALL_PACKAGES': 'uses_QUERY_ALL_PACKAGES'
            }
            
            # Set permission features based on manifest
            for perm in permissions:
                if perm in permission_mapping:
                    permission_features[permission_mapping[perm]] = 1
            
            # Check for specific API usage in smali code
            api_usage_count = self._check_api_usage_in_smali(smali_dir)
            
            # Set API-specific features
            if api_usage_count.get('getDeviceId', 0) > 0:
                permission_features['uses_getDeviceId'] = 1
            if api_usage_count.get('getSubscriberId', 0) > 0:
                permission_features['uses_getSubscriberId'] = 1
            
            # Calculate num_sensitive_flags
            permission_features['num_sensitive_flags'] = sum(permission_features.values())
            
            print(f"Permission features extracted: {permission_features['num_sensitive_flags']} sensitive flags found")
            
        except Exception as e:
            print(f"Error extracting permission features: {e}")
            # Return default values if extraction fails
            for perm in self.permission_features:
                permission_features[perm] = 0
            permission_features['num_sensitive_flags'] = 0
        
        return permission_features
    
    def _check_api_usage_in_smali(self, smali_dir):
        """Check for specific API usage in smali files."""
        api_usage = {
            'getDeviceId': 0,
            'getSubscriberId': 0
        }
        
        try:
            for root, _, files in os.walk(smali_dir):
                for file in files:
                    if file.endswith('.smali'):
                        filepath = os.path.join(root, file)
                        try:
                            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                if 'getDeviceId' in content:
                                    api_usage['getDeviceId'] += content.count('getDeviceId')
                                if 'getSubscriberId' in content:
                                    api_usage['getSubscriberId'] += content.count('getSubscriberId')
                        except:
                            continue
        except Exception as e:
            print(f"Error checking API usage: {e}")
        
        return api_usage
    
    def _extract_dex(self, apk_path, output_dir):
        """Extract DEX file from APK."""
        try:
            print("Extracting DEX file...")
            apk = APK(apk_path)
            
            if not apk.is_valid_APK():
                print("Invalid APK file")
                return None
            
            dex_path = os.path.join(output_dir, "classes.dex")
            dex_data = apk.get_dex()
            
            if dex_data:
                with open(dex_path, "wb") as f:
                    f.write(dex_data)
                print(f"DEX extracted to: {dex_path}")
                return dex_path
            else:
                print("No DEX data found in APK")
                return None
                
        except Exception as e:
            print(f"Error extracting DEX: {e}")
            return None
    
    def _extract_smali(self, dex_path, output_dir):
        """Convert DEX to smali code using baksmali."""
        try:
            print("Converting DEX to smali...")
            smali_dir = os.path.join(output_dir, "smali")
            os.makedirs(smali_dir, exist_ok=True)
            
            if not os.path.exists(self.baksmali_jar):
                print(f"Baksmali JAR not found: {self.baksmali_jar}")
                return None
            
            command = f'java -jar "{self.baksmali_jar}" d "{dex_path}" -o "{smali_dir}"'
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Smali code extracted to: {smali_dir}")
                return smali_dir
            else:
                print(f"Baksmali failed: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"Error extracting smali: {e}")
            return None
    
    def _extract_structural_features(self, smali_dir):
        """Extract structural features from smali code."""
        print("Extracting structural features...")
        
        # Parse smali files and build call graph
        call_graph, method_info = self._build_call_graph(smali_dir)
        
        # Calculate graph metrics
        features = {}
        
        # Basic counts
        features['total_methods'] = len(method_info)
        features['total_edges'] = call_graph.number_of_edges()
        features['sensitive_apis'] = self._count_sensitive_apis(method_info)
        
        # Graph density
        if len(call_graph) > 1:
            features['graph_density'] = nx.density(call_graph)
        else:
            features['graph_density'] = 0.0
        
        # Degree statistics
        if len(call_graph) > 0:
            in_degrees = [d for n, d in call_graph.in_degree()]
            out_degrees = [d for n, d in call_graph.out_degree()]
            total_degrees = [call_graph.degree(n) for n in call_graph.nodes()]
            
            # In-degree stats
            features['avg_in_degree'] = np.mean(in_degrees) if in_degrees else 0
            features['max_in_degree'] = np.max(in_degrees) if in_degrees else 0
            features['min_in_degree'] = np.min(in_degrees) if in_degrees else 0
            features['std_in_degree'] = np.std(in_degrees) if in_degrees else 0
            
            # Out-degree stats
            features['avg_out_degree'] = np.mean(out_degrees) if out_degrees else 0
            features['max_out_degree'] = np.max(out_degrees) if out_degrees else 0
            features['min_out_degree'] = np.min(out_degrees) if out_degrees else 0
            features['std_out_degree'] = np.std(out_degrees) if out_degrees else 0
            
            # Total degree stats
            features['avg_total_degree'] = np.mean(total_degrees) if total_degrees else 0
            features['max_total_degree'] = np.max(total_degrees) if total_degrees else 0
            features['min_total_degree'] = np.min(total_degrees) if total_degrees else 0
            features['std_total_degree'] = np.std(total_degrees) if total_degrees else 0
            
            # Centrality measures
            try:
                degree_centrality = list(nx.degree_centrality(call_graph).values())
                features['avg_degree_centrality'] = np.mean(degree_centrality) if degree_centrality else 0
                features['max_degree_centrality'] = np.max(degree_centrality) if degree_centrality else 0
                features['min_degree_centrality'] = np.min(degree_centrality) if degree_centrality else 0
                features['std_degree_centrality'] = np.std(degree_centrality) if degree_centrality else 0
                
                pagerank = list(nx.pagerank(call_graph).values())
                features['avg_pagerank'] = np.mean(pagerank) if pagerank else 0
                features['max_pagerank'] = np.max(pagerank) if pagerank else 0
                features['min_pagerank'] = np.min(pagerank) if pagerank else 0
                features['std_pagerank'] = np.std(pagerank) if pagerank else 0
            except:
                # If centrality calculation fails, set to zero
                features.update({
                    'avg_degree_centrality': 0, 'max_degree_centrality': 0,
                    'min_degree_centrality': 0, 'std_degree_centrality': 0,
                    'avg_pagerank': 0, 'max_pagerank': 0,
                    'min_pagerank': 0, 'std_pagerank': 0
                })
        else:
            # Empty graph - set all metrics to zero
            zero_metrics = [
                'avg_in_degree', 'max_in_degree', 'min_in_degree', 'std_in_degree',
                'avg_out_degree', 'max_out_degree', 'min_out_degree', 'std_out_degree',
                'avg_total_degree', 'max_total_degree', 'min_total_degree', 'std_total_degree',
                'avg_degree_centrality', 'max_degree_centrality', 'min_degree_centrality', 'std_degree_centrality',
                'avg_pagerank', 'max_pagerank', 'min_pagerank', 'std_pagerank'
            ]
            for metric in zero_metrics:
                features[metric] = 0.0
        
        return features
    
    def _build_call_graph(self, smali_dir):
        """Build call graph from smali files."""
        call_graph = nx.DiGraph()
        method_info = {}
        
        for root, _, files in os.walk(smali_dir):
            for file in files:
                if file.endswith('.smali'):
                    filepath = os.path.join(root, file)
                    self._parse_smali_file(filepath, call_graph, method_info)
        
        return call_graph, method_info
    
    def _parse_smali_file(self, filepath, call_graph, method_info):
        """Parse a single smali file and extract method calls."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                current_method = None
                current_class = None
                
                for line in f:
                    line = line.strip()
                    
                    # Extract class name
                    if line.startswith('.class'):
                        current_class = line.split()[-1]
                    
                    # Extract method definition
                    elif line.startswith('.method'):
                        method_parts = line.split()
                        if len(method_parts) > 2:
                            current_method = f"{current_class}->{method_parts[-1]}"
                            method_info[current_method] = {
                                'class': current_class,
                                'calls': [],
                                'is_sensitive': False
                            }
                            call_graph.add_node(current_method)
                    
                    # Extract method calls
                    elif line.startswith('invoke-') and current_method:
                        try:
                            # Extract the called method
                            call_part = line.split(',')[-1].strip()
                            if '->' in call_part:
                                called_method = call_part.split('->')[0] + '->' + call_part.split('->')[1].split('(')[0]
                                method_info[current_method]['calls'].append(called_method)
                                call_graph.add_edge(current_method, called_method)
                                
                                # Check if it's a sensitive API
                                if any(sensitive in called_method for sensitive in self.sensitive_apis):
                                    method_info[current_method]['is_sensitive'] = True
                        except:
                            pass
        except Exception as e:
            print(f"Warning: Error parsing {filepath}: {e}")
    
    def _count_sensitive_apis(self, method_info):
        """Count methods that call sensitive APIs."""
        return sum(1 for method in method_info.values() if method['is_sensitive'])
    
    def _extract_semantic_features(self, smali_dir):
        """Extract semantic features using Word2Vec embeddings."""
        print("Extracting semantic features...")
        
        # Tokenize smali code
        tokens = self._tokenize_smali_code(smali_dir)
        
        if not tokens:
            print("No tokens found for semantic features")
            return {str(i): 0.0 for i in range(100)}  # Return 100 zero features
        
        try:
            # Train Word2Vec model
            model = Word2Vec(
                sentences=[tokens],
                vector_size=100,
                window=5,
                min_count=1,
                workers=4,
                sg=1  # Skip-gram
            )
            
            # Get document vector by averaging word vectors
            vectors = []
            for token in tokens:
                if token in model.wv:
                    vectors.append(model.wv[token])
            
            if vectors:
                doc_vector = np.mean(vectors, axis=0)
            else:
                doc_vector = np.zeros(100)
            
            # Convert to dictionary with string keys (0-99)
            semantic_features = {str(i): float(value) for i, value in enumerate(doc_vector)}
            
            return semantic_features
            
        except Exception as e:
            print(f"Error generating semantic features: {e}")
            return {str(i): 0.0 for i in range(100)}
    
    def _tokenize_smali_code(self, smali_dir):
        """Extract tokens from smali files."""
        tokens = []
        
        for root, _, files in os.walk(smali_dir):
            for file in files:
                if file.endswith('.smali'):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            for line in f:
                                # Simple tokenization
                                line_tokens = line.strip().split()
                                tokens.extend(line_tokens)
                    except Exception as e:
                        continue
        
        return tokens
    
    def _predict(self, features, apk_name):
        """Make prediction using the trained model."""
        print("Making prediction...")
        
        # Create feature vector matching training data format
        feature_vector = []
        missing_features = []
        
        for feature_name in self.feature_names:
            if feature_name == 'apk_name':
                continue  # Skip non-numeric feature
            
            if feature_name in features:
                feature_vector.append(features[feature_name])
            else:
                feature_vector.append(0.0)  # Default value for missing features
                missing_features.append(feature_name)
        
        if missing_features:
            print(f"Missing features (set to 0): {len(missing_features)} features")
        
        # Convert to numpy array and reshape
        feature_array = np.array(feature_vector).reshape(1, -1)
        
        # Scale features
        feature_array_scaled = self.scaler.transform(feature_array)
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(feature_array_scaled)
        malware_probability = probabilities[0][1]  # Probability of being malicious
      

        prediction = 1 if malware_probability >= 0. else 0
        
        print(f"Prediction: {'Malicious' if prediction == 1 else 'Benign'} ({malware_probability:.2%})")
        
        return prediction, malware_probability
    
    def _generate_shap_explanation(self, features, apk_name):
        """Generate SHAP explanations for the prediction."""
        print("🔍 Generating SHAP explanations...")

        try:
            # Prepare feature vector
            feature_vector = []
            feature_names_for_shap = []

            for feature_name in self.feature_names:
                if feature_name == 'apk_name':
                    continue
                feature_vector.append(features.get(feature_name, 0.0))
                feature_names_for_shap.append(feature_name)

            # Reshape and scale the feature vector
            feature_array = np.array(feature_vector).reshape(1, -1)
            feature_array_scaled = self.scaler.transform(feature_array)

            # Initialize SHAP KernelExplainer with proper background
            if self.explainer is None:
                print("🚀 Initializing SHAP KernelExplainer...")

                # Recommended: load a background sample from training data
                # background = pd.read_csv("models/shap_background_sample.csv").values

                # Or use repeated sample as approximate background (not ideal but works)
                background = np.repeat(feature_array_scaled, 10, axis=0)

                self.explainer = shap.KernelExplainer(self.model.predict_proba, background)

            # Compute SHAP values
            shap_values = self.explainer.shap_values(feature_array_scaled)

            # Handle binary classifier (VotingClassifier returns list)
            if isinstance(shap_values, list):
                shap_vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            else:
                shap_vals = shap_values[0]

            explanation = {
                'shap_values': shap_vals.tolist(),
                'feature_names': feature_names_for_shap,
                'feature_values': feature_vector,
                'base_value': float(self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value)
            }

            # Compute top 10 important features
            feature_importance = []
            for name, value, shap_val in zip(feature_names_for_shap, feature_vector, shap_vals):
                feature_importance.append({
                    'feature': name,
                    'value': float(value),
                    'shap_value': float(shap_val),
                    'importance': abs(float(shap_val))
                })

            feature_importance.sort(key=lambda x: x['importance'], reverse=True)
            explanation['top_10_features'] = feature_importance[:10]

            print(f"✅ SHAP explanation generated with {len(shap_vals)} features")
            return explanation

        except Exception as e:
            print(f"⚠️ Error generating SHAP explanation: {e}")
            return {
                'error': str(e),
                'shap_values': [],
                'top_10_features': []
            }

    def _get_feature_summary(self, features):
        """Get a summary of extracted features."""
        summary = {
                'structural_features': 0,
                'semantic_features': 0,
                'permission_features': 0,
                'sensitive_apis': features.get('sensitive_apis', 0),
                'total_methods': features.get('total_methods', 0),
                'graph_density': features.get('graph_density', 0.0)
            }
            
            # Count different types of features
        for feature_name in features.keys():
            if feature_name.isdigit():  # Semantic features (Word2Vec dimensions)
                summary['semantic_features'] += 1
            elif feature_name.startswith('uses_') or feature_name == 'num_sensitive_flags':
                summary['permission_features'] += 1
            else:
                summary['structural_features'] += 1
            
        return summary

    def _print_classification_summary(self, result):
        """Print a formatted summary of the classification results."""
        print("\n" + "="*80)
        print("CLASSIFICATION SUMMARY")
        print("="*80)
        print(f"APK Name: {result['apk_name']}")
        print(f"Prediction: {result['prediction']}")
        #print(f"Confidence: {result['confidence']:.2%}")
        print(f"Features Extracted: {result['features_extracted']}")
        
        if 'feature_summary' in result:
            summary = result['feature_summary']
            print(f"Structural Features: {summary['structural_features']}")
            print(f"Semantic Features: {summary['semantic_features']}")
            print(f"Permission Features: {summary['permission_features']}")
            print(f"Sensitive APIs: {summary['sensitive_apis']}")
            print(f"Total Methods: {summary['total_methods']}")
            print(f"Graph Density: {summary['graph_density']:.4f}")
        
        # Print top SHAP features
        if 'shap_explanation' in result and 'top_10_features' in result['shap_explanation']:
            #print("\nTOP 10 MOST IMPORTANT FEATURES:")
            print("-" * 50)
            for i, feature in enumerate(result['shap_explanation']['top_10_features'][:5], 1):
                impact = "Malicious" if feature['shap_value'] > 0 else "Benign"
                print(f"{i:2d}. {feature['feature'][:40]:40s} | {impact} | {feature['importance']:.4f}")
        
        if 'visualization_paths' in result:
            print(f"\nVisualizations saved to: {len(result['visualization_paths'])} files")
            for viz_type, path in result['visualization_paths'].items():
                print(f"   • {viz_type}: {path}")
        
        print("="*80)

def classify_multiple_apks(self, apk_paths, max_workers=4):
    """
    Classify multiple APK files in parallel.
    
    Args:
        apk_paths: List of APK file paths
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of classification results
    """
    print(f"Starting batch classification of {len(apk_paths)} APK files...")
    
    results = []
    
    def classify_single(apk_path):
        try:
            return self.classify_apk(apk_path)
        except Exception as e:
            return {
                "apk_name": os.path.basename(apk_path),
                "error": f"Classification failed: {str(e)}"
            }
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_apk = {executor.submit(classify_single, apk_path): apk_path 
                        for apk_path in apk_paths}
        
        for future in future_to_apk:
            try:
                result = future.result(timeout=300)  # 5 minute timeout per APK
                results.append(result)
                
                if 'error' not in result:
                    print(f"Completed: {result['apk_name']} - {result['prediction']}")
                else:
                    print(f"Failed: {result['apk_name']} - {result['error']}")
                    
            except Exception as e:
                apk_path = future_to_apk[future]
                results.append({
                    "apk_name": os.path.basename(apk_path),
                    "error": f"Processing timeout or error: {str(e)}"
                })
                print(f"Timeout/Error: {os.path.basename(apk_path)}")
    
    # Print batch summary
    successful = sum(1 for r in results if 'error' not in r)
    malicious = sum(1 for r in results if 'error' not in r and r['prediction'] == 'Malicious')
    
    print(f"\nBATCH CLASSIFICATION SUMMARY:")
    print(f"Total APKs: {len(apk_paths)}")
    print(f"Successfully classified: {successful}")
    print(f"Malicious: {malicious}")
    print(f"Benign: {successful - malicious}")
    print(f"Failed: {len(apk_paths) - successful}")
    
    return results

def save_results_to_csv(self, results, output_path="classification_results.csv"):
    """Save classification results to CSV file."""
    print(f"Saving results to {output_path}...")
    
    # Prepare data for CSV
    csv_data = []
    for result in results:
        if 'error' not in result:
            row = {
                'apk_name': result['apk_name'],
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'features_extracted': result['features_extracted']
            }
            
            # Add feature summary
            if 'feature_summary' in result:
                summary = result['feature_summary']
                row.update({
                    'structural_features': summary['structural_features'],
                    'semantic_features': summary['semantic_features'],
                    'permission_features': summary['permission_features'],
                    'sensitive_apis': summary['sensitive_apis'],
                    'total_methods': summary['total_methods'],
                    'graph_density': summary['graph_density']
                })
            
            # Add top SHAP features
            if ('shap_explanation' in result and 
                'top_10_features' in result['shap_explanation']):
                top_features = result['shap_explanation']['top_10_features'][:5]
                for i, feature in enumerate(top_features, 1):
                    row[f'top_feature_{i}'] = feature['feature']
                    row[f'top_feature_{i}_importance'] = feature['importance']
            
            csv_data.append(row)
        else:
            # Add error entries
            csv_data.append({
                'apk_name': result['apk_name'],
                'prediction': 'ERROR',
                'confidence': 0.0,
                'error': result['error']
            })
    
    # Create DataFrame and save
    df = pd.DataFrame(csv_data)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


# Example usage and main function
def main():
    """Main function to demonstrate usage."""
    print("APK Malware Classifier")
    print("=" * 50)
    
    # Initialize classifier
    try:
        classifier = APKClassifier()
        print("Classifier initialized successfully")
    except Exception as e:
        print(f"Failed to initialize classifier: {e}")
        return
    
    # Example: Classify a single APK
    apk_path = r"D:\VScode programs\Mini\testing\benign\000ACB08C76FFC86C18543573ACA14C29A5A59749A114B0B0E3CEC6FD9D4677E.apk"  # Replace with actual APK path
    if os.path.exists(apk_path):
        result = classifier.classify_apk(apk_path)
        
        if 'error' not in result:
            print(f"\nClassification Result:")
            print(f"APK: {result['apk_name']}")
            print(f"Prediction: {result['prediction']}")
            #print(f"Confidence: {result['confidence']:.2%}")
        else:
            print(f"Classification failed: {result['error']}")
    
    # Example: Batch classification
    apk_directory = "apk_samples"  # Replace with actual directory
    if os.path.exists(apk_directory):
        apk_files = [os.path.join(apk_directory, f) 
                    for f in os.listdir(apk_directory) 
                    if f.endswith('.apk')]
        
        if apk_files:
            print(f"\nProcessing {len(apk_files)} APK files...")
            results = classifier.classify_multiple_apks(apk_files)
            
            # Save results
            classifier.save_results_to_csv(results)
            print("Batch processing completed")


if __name__ == "__main__":
    main()