import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
import requests
import json
import time
import re
from typing import Dict, Any
import warnings
import httpx
import chardet

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

os.environ['AIPROXY_TOKEN'] = 'eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjEwMDI5NDBAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.Kymefw6W0esqnKNCPMT2njCfj6bNi7c-AsVhJD3R7n8'


def do_generic_analysis(df: pd.DataFrame, cluster_num: int = 3, outlier_z: float = 3.0) -> Dict[str, Any]:
    """
    Perform generic analysis on a pandas DataFrame including summary statistics,
    missing values count, correlation matrix, outlier detection, clustering,
    PCA, and hierarchical clustering.

    Parameters:
    - df: pandas.DataFrame
        The input dataset.
    - cluster_num: int, default=3
        Number of clusters for K-Means.
    - outlier_z: float, default=3.0
        Z-score threshold for outlier detection.

    Returns:
    - analysis_results: dict
        A dictionary containing various analysis results.
    """
    analysis_results = {}
    
    # 1. Summary Statistics
    summary = df.describe(include='all').transpose()
    analysis_results['summary_statistics'] = summary
    
    # 2. Missing Values
    missing_values = df.isnull().sum()
    missing_percent = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage (%)': missing_percent
    })
    analysis_results['missing_values'] = missing_df
    
    # 3. Correlation Matrix
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        corr_matrix = numeric_df.corr()
        analysis_results['correlation_matrix'] = corr_matrix
    else:
        analysis_results['correlation_matrix'] = pd.DataFrame()
    
    # 4. Outlier Detection
    outliers = {}
    for col in numeric_df.columns:
        if numeric_df[col].nunique() > 1:  # Avoid division by zero in zscore
            z_scores = np.abs(stats.zscore(numeric_df[col].dropna()))
            outlier_indices = numeric_df[col].dropna().index[z_scores > outlier_z].tolist()
            outliers[col] = outlier_indices
    analysis_results['outliers'] = outliers
    
    # 5. Clustering
    # Preprocessing: Handle missing values and encode categorical variables
    processed_df = df.copy()
    
    # Fill numerical missing values with mean
    for col in numeric_df.columns:
        processed_df[col].fillna(processed_df[col].mean(), inplace=True)
    
    # Encode categorical variables
    categorical_cols = processed_df.select_dtypes(include=['object', 'category']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        processed_df[col] = processed_df[col].astype(str)
        processed_df[col].fillna('Missing', inplace=True)
        processed_df[col] = le.fit_transform(processed_df[col])
    
    # Scaling
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(processed_df)
    
    # K-Means Clustering
    kmeans = KMeans(n_clusters=cluster_num, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    processed_df['Cluster'] = clusters
    analysis_results['kmeans_clusters'] = pd.Series(clusters)
    
    # PCA for Visualization (Textual Summary)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)
    explained_variance = pca.explained_variance_ratio_
    analysis_results['pca_explained_variance'] = explained_variance
    
    # 6. Hierarchical Clustering
    linked = linkage(scaled_features, method='ward')
    analysis_results['hierarchical_linkage'] = linked
    
    return analysis_results

def generate_correlation_heatmap(corr_matrix: pd.DataFrame, output_path: str):
    """
    Generate and save a correlation heatmap.

    Parameters:
    - corr_matrix: pandas.DataFrame
        The correlation matrix.
    - output_path: str
        The file path to save the heatmap.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def generate_pca_scatter(principal_components: np.ndarray, clusters: pd.Series, explained_variance: np.ndarray, output_path: str):
    """
    Generate and save a PCA scatter plot colored by cluster assignments.

    Parameters:
    - principal_components: numpy.ndarray
        The principal components obtained from PCA.
    - clusters: pandas.Series
        Cluster assignments from K-Means.
    - explained_variance: numpy.ndarray
        The explained variance ratio from PCA.
    - output_path: str
        The file path to save the PCA scatter plot.
    """
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], c=clusters, cmap='viridis')
    plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
    plt.title('PCA Scatter Plot with K-Means Clusters')
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def generate_dendrogram(linked: np.ndarray, labels: list, output_path: str):
    """
    Generate and save a hierarchical clustering dendrogram.

    Parameters:
    - linked: numpy.ndarray
        The linkage matrix from hierarchical clustering.
    - labels: list
        Labels for the dendrogram.
    - output_path: str
        The file path to save the dendrogram.
    """
    plt.figure(figsize=(10, 7))
    dendrogram(linked, labels=labels, leaf_rotation=90)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def analyze_with_openai(filename: str, df: pd.DataFrame, analysis_results: Dict[str, Any], output_dir: str, additional_context: str = None) -> Dict[str, Any]:
    """
    Interact with OpenAI API to analyze the data based on generic analysis.
    This function requests suggestions for additional analyses and Python code snippets
    from the LLM, executes the code with caution, and provides enhanced insights.

    Parameters:
    - filename: str
        The name of the dataset file.
    - df: pandas.DataFrame
        The input dataset.
    - analysis_results: dict
        The dictionary containing results from do_generic_analysis.
    - output_dir: str
        The directory path to save all generated images and charts.
    - additional_context: str, optional
        Any additional context or information about the dataset.

    Returns:
    - enhanced_results: dict
        A dictionary containing all analysis results, including those from OpenAI.
    """
    # Retrieve API key from environment variable
    openai_api_key = os.getenv('AIPROXY_TOKEN')
    if not openai_api_key:
        print("Error: OpenAI API key not found in 'AIPROXY_TOKEN' environment variable.")
        return analysis_results  # Return existing analysis results
    
    # Set OpenAI API configuration
    api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

    # Prepare the prompt with dataset details
    column_info = "\n".join([f"- {col}: {df[col].dtype}" for col in df.columns])
    summary_stats = analysis_results.get('summary_statistics', pd.DataFrame()).to_string()
    missing_values = analysis_results.get('missing_values', pd.DataFrame()).to_string()
    correlation_matrix = analysis_results.get('correlation_matrix', pd.DataFrame()).to_string()
    
    # Outliers: summarize number of outliers per column
    outliers = analysis_results.get('outliers', {})
    outlier_summary = "\n".join([f"- {col}: {len(indices)} outliers" for col, indices in outliers.items()])
    
    # Cluster distribution
    clusters = analysis_results.get('kmeans_clusters', [])
    if isinstance(clusters, pd.Series):
        cluster_counts = clusters.value_counts().sort_index().to_dict()
    else:
        cluster_counts = pd.Series(clusters).value_counts().sort_index().to_dict()
    
    # PCA explained variance
    pca_variance = analysis_results.get('pca_explained_variance', [])
    pca_info = ""
    if len(pca_variance) >= 2:
        pca_info = f"PC1: {pca_variance[0]:.2%}, PC2: {pca_variance[1]:.2%}"
    elif len(pca_variance) == 1:
        pca_info = f"PC1: {pca_variance[0]:.2%}"
    else:
        pca_info = "No PCA variance information available."
    
    # Enhanced Prompt with Explicit Instructions to Save Charts
    prompt = f"""
You are an expert data analyst. Analyze the following dataset information and provide insights.

**Filename**: {filename}

**Columns and Types**:
{column_info}

**Summary Statistics**:
{summary_stats}

**Missing Values**:
{missing_values}

**Correlation Matrix**:
{correlation_matrix}

**Outliers**:
{outlier_summary}

**K-Means Clusters**:
Cluster distribution: {cluster_counts}

**PCA Explained Variance**:
{pca_info}

**Hierarchical Clustering**:
Linkage matrix computed.

**Additional Context**:
{additional_context if additional_context else "None"}

Based on the above information, please perform the following tasks:

1. **Suggest Additional Analyses**: Recommend specific analyses or function calls that could yield more insights into the data.

2. **Provide Python Code Snippets**: For each suggested analysis, provide Python code that can be executed to perform the analysis. Ensure that the code utilizes the existing `df` DataFrame and does not reload the dataset. **Additionally, ensure that any plots or charts generated by the code are saved into the `output_dir`. Use `os.path.join(output_dir, 'chart_name.png')` to define the file paths for saving images.**

Ensure that the code is syntactically correct and handles potential errors gracefully.
"""

    # Define a function to handle retries
    def call_openai_api_with_retries(prompt, max_retries=3, backoff_factor=2):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }
        data = {
            "model": "gpt-4o-mini",  
            "messages": [
                {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1500,
            "temperature": 0.7
        }

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(api_url, headers=headers, data=json.dumps(data), timeout=60)
                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"Attempt {attempt}: Received status code {response.status_code} with message: {response.text}")
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt} failed with error: {e}")

            if attempt < max_retries:
                sleep_time = backoff_factor ** attempt
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("All retry attempts failed.")
                return None

    # Call the API with retries
    response_json = call_openai_api_with_retries(prompt)
    if not response_json:
        print("Failed to get a response from OpenAI API.")
        return analysis_results  # Return existing analysis results

    # Extract the answer from the response
    try:
        answer = response_json['choices'][0]['message']['content'].strip()
    except (KeyError, IndexError) as e:
        print(f"Error parsing response JSON: {e}")
        print("Full response:", response_json)
        return analysis_results  # Return existing analysis results

    print("\n=== OpenAI Analysis ===\n")
    print(answer)
    print("\n" + "="*50 + "\n")

    # Initialize enhanced_results with existing analysis_results
    enhanced_results = analysis_results.copy()

    # Parse the answer to extract different sections
    # Assuming the LLM formats sections with headers like ### Additional Analyses
    suggestions_match = re.search(r'###\s*Additional Analyses\s*([\s\S]*?)(?:\n###|$)', answer, re.IGNORECASE)
    code_snippets = re.findall(r'```python\n(.*?)```', answer, re.DOTALL)

    if suggestions_match:
        suggestions_text = suggestions_match.group(1).strip()
        # Assume suggestions are in a list format
        suggestions = re.findall(r'\d+\.\s*(.+)', suggestions_text)
        enhanced_results['openai_suggestions'] = suggestions
        print("=== OpenAI Suggestions for Additional Analyses ===")
        for idx, suggestion in enumerate(suggestions, 1):
            print(f"{idx}. {suggestion}")
        print("\n" + "="*50 + "\n")
    else:
        # Fallback if specific sections are not found
        enhanced_results['openai_suggestions'] = []
        print("No specific suggestions found from OpenAI.\n" + "="*50 + "\n")

    # Execute each code snippet and capture results
    for idx, code in enumerate(code_snippets, 1):
        print(f"--- Executing Code Snippet {idx} ---")
        try:
            # Define a local namespace for code execution
            local_namespace = {
                'df': df,
                'analysis_results': enhanced_results,
                'output_dir': output_dir,
                'os': os,
                'plt': plt,
                'sns': sns,
                'pd': pd,
                'np': np,
                'StandardScaler': StandardScaler,
                'LabelEncoder': LabelEncoder,
                'PCA': PCA,
                'KMeans': KMeans,
                'stats': stats,
                'linkage': linkage,
                'dendrogram': dendrogram
            }
            exec(code, globals(), local_namespace)
            print(f"Code Snippet {idx} executed successfully.\n")
            # Retrieve any updates made to 'analysis_results' or 'enhanced_results'
            if 'analysis_results' in local_namespace:
                for key, value in local_namespace['analysis_results'].items():
                    enhanced_results[key] = value
            if 'enhanced_results' in local_namespace:
                for key, value in local_namespace['enhanced_results'].items():
                    enhanced_results[key] = value
        except Exception as e:
            print(f"Error executing Code Snippet {idx}: {e}\n")
    print("OpenAI analysis and code execution completed.\n" + "="*50 + "\n")

    return enhanced_results

#---------------------------STORY--------------------------
import os
import requests
import pandas as pd
from PIL import Image, __version__ as PILLOW_VERSION
import io
import base64
import re

def get_resampling_filter():
    """
    Returns the appropriate resampling filter based on the Pillow version.
    """
    # Extract major and minor version numbers
    version_match = re.match(r"(\d+)\.(\d+)", PILLOW_VERSION)
    if version_match:
        major, minor = map(int, version_match.groups())
        if major >= 10:
            return Image.LANCZOS  # For Pillow 10.0.0 and above
    return Image.ANTIALIAS  # For older versions

def encode_image_to_data_uri(image_path):
    """
    Encodes an image to a Base64 Data URI.
    
    Parameters:
    - image_path (str): Path to the image file.
    
    Returns:
    - str: Data URI of the encoded image.
    """
    try:
        with Image.open(image_path) as img:
            # Convert image to RGB if it's in a different mode
            img = img.convert("RGB")
            # Resize image to lower resolution (e.g., 256x256)
            resample_filter = get_resampling_filter()
            img = img.resize((300, 300), resample=resample_filter)
            
            # Save the image to a bytes buffer with reduced quality
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=50)  # Adjust quality as needed
            buffered.seek(0)
            
            # Encode the image in Base64
            img_base64 = base64.b64encode(buffered.read()).decode('utf-8')
            
            # Create Data URI
            data_uri = f"data:image/jpeg;base64,{img_base64}"
            return data_uri
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def generate_story(df, output_loc, analysis):
    """
    Generates a narrative story based on the analysis and images provided,
    and saves the output as README.md in the output_loc directory.

    Parameters:
    - df (pandas.DataFrame): Input data frame.
    - output_loc (str): Path to the directory containing images.
    - analysis (str): Analysis containing summary statistics and more.

    Returns:
    - None
    """
    # Retrieve API key from environment variable
    openai_api_key = os.getenv('AIPROXY_TOKEN')
    if not openai_api_key:
        raise EnvironmentError("API key not found in environment variable 'AIPROXY_TOKEN'.")

    # Set OpenAI API configuration
    api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    # Ensure output_loc exists
    if not os.path.isdir(output_loc):
        raise ValueError(f"The output location '{output_loc}' does not exist or is not a directory.")

    # Supported image formats
    supported_formats = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')

    # Retrieve image filenames
    image_files = [f for f in os.listdir(output_loc) if f.lower().endswith(supported_formats)]

    images_info = ""
    if not image_files:
        print("No image files found in the specified output directory.")
    else:
        # Encode images as Data URIs
        data_uris = []
        for image_file in image_files:
            image_path = os.path.join(output_loc, image_file)
            data_uri = encode_image_to_data_uri(image_path)
            if data_uri:
                # Create Markdown image link with Data URI
                markdown_image = f"![{os.path.splitext(image_file)[0]}]({data_uri})"
                data_uris.append(markdown_image)
            else:
                print(f"Skipping image {image_file} due to encoding failure.")

        if data_uris:
            images_info = "\n\n".join(data_uris)
        else:
            print("No images were successfully encoded.")

    # Construct the prompt with instruction about low-detail images
    prompt = (
        "Narrate a story based on the analysis and low-detail images provided. Describe:\n\n"
        "1. The data you received, briefly.\n"
        "2. The analysis you carried out.\n"
        "3. The insights you discovered.\n"
        "4. The implications of your findings (i.e., what to do with the insights).\n\n"
    )

    if images_info:
        prompt += (
            "Incorporate the following low-detail images if relavent into the narrative using Markdown image links:\n\n"
            f"{images_info}\n\n"
        )

    prompt += "Provide the output in a neatly formatted Markdown file."

    # Prepare the data payload without the 'detail' parameter
    data = {
        "model": "gpt-4o-mini",  
        "messages": [
            {"role": "system", "content": "You are a knowledgeable data analyst and a skilled storyteller."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": f"### Analysis Details:\n{analysis}"}
        ],
        "max_tokens": 1500,
        "temperature": 0.7
    }

    try:
        # Make the API request using the proxy
        response = requests.post(api_url, headers=headers, json=data)

        # Check if the request was successful
        response.raise_for_status()

        # Parse the JSON response
        response_json = response.json()

        # Extract the generated story
        story = response_json['choices'][0]['message']['content'].strip()

        # Define the path for README.md
        readme_path = os.path.join(output_loc, 'README.md')

        # Save the story to README.md
        with open(readme_path, 'w', encoding='utf-8') as file:
            file.write(story)

        print(f"README.md has been successfully created at: {readme_path}")

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - Response: {response.text}")
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
    except KeyError:
        print("Unexpected response structure. 'choices' or 'message' keys not found.")
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")




def main():
    # Argument Parsing
    parser = argparse.ArgumentParser(description='Automated Exploratory Data Analysis (EDA) Script')
    parser.add_argument('csv_file', type=str, help='Path to the CSV dataset file')
    args = parser.parse_args()

    csv_file = args.csv_file

    # Validate CSV file existence
    if not os.path.isfile(csv_file):
        print(f"Error: File '{csv_file}' does not exist.")
        sys.exit(1)

    # Extract dataset name (without extension) for the output directory
    dataset_name = os.path.splitext(os.path.basename(csv_file))[0]
    output_dir = dataset_name

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    else:
        print(f"Directory already exists: {output_dir}")


    # Load the dataset
    with open(csv_file, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']
    print(f"Detected file encoding: {encoding}")

    try:
        df = pd.read_csv(csv_file, encoding=encoding)
        print(f"Loaded dataset '{csv_file}' successfully.\n")
    except Exception as e:
        print(f"Error loading '{csv_file}': {e}")
        sys.exit(1)

    # Perform generic analysis
    print("Performing generic exploratory data analysis...\n")
    analysis_results = do_generic_analysis(df, cluster_num=3, outlier_z=3.0)
    print("Generic analysis completed.\n")

    # Generate and save Correlation Heatmap
    if not analysis_results['correlation_matrix'].empty:
        heatmap_path = os.path.join(output_dir, 'correlation_heatmap.png')
        generate_correlation_heatmap(analysis_results['correlation_matrix'], heatmap_path)
        print(f"Correlation heatmap saved to {heatmap_path}\n")
    else:
        print("No numerical columns available for correlation heatmap.\n")

    # Generate and save PCA Scatter Plot
    if 'pca_explained_variance' in analysis_results and len(analysis_results['pca_explained_variance']) >= 2:
        # Recompute PCA to get principal components for plotting
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        # Fill NaN values with mean
        numeric_df.fillna(numeric_df.mean(), inplace=True)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(numeric_df)
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(scaled_features)
        explained_variance = pca.explained_variance_ratio_
        pca_path = os.path.join(output_dir, 'pca_scatter.png')
        generate_pca_scatter(
            principal_components=principal_components,
            clusters=analysis_results['kmeans_clusters'],
            explained_variance=explained_variance,
            output_path=pca_path
        )
        print(f"PCA scatter plot saved to {pca_path}\n")
    else:
        print("Insufficient PCA variance information for scatter plot.\n")

    # Generate and save Hierarchical Clustering Dendrogram
    if 'hierarchical_linkage' in analysis_results and analysis_results['hierarchical_linkage'] is not None:
        dendrogram_path = os.path.join(output_dir, 'hierarchical_dendrogram.png')
        labels = df.index.tolist()
        generate_dendrogram(analysis_results['hierarchical_linkage'], labels, dendrogram_path)
        print(f"Hierarchical clustering dendrogram saved to {dendrogram_path}\n")
    else:
        print("No hierarchical linkage information available for dendrogram.\n")

    # Optional: OpenAI Integration for Advanced Analysis
    # Check if OpenAI API key is set
    if os.getenv('AIPROXY_TOKEN'):
        print("Initiating OpenAI-based advanced analysis...\n")
        enhanced_results = analyze_with_openai(
            filename=csv_file,
            df=df,
            analysis_results=analysis_results,
            output_dir=output_dir,
            additional_context="This dataset contains data from Goodreads."
        )
        print("Advanced analysis completed.\n")
    else:
        print("OpenAI API key not found. Skipping advanced analysis.\n")

    # Save analysis results to a JSON file
    results_path = os.path.join(output_dir, 'analysis_results.json')
    try:
        # Convert all non-serializable objects to serializable formats
        serializable_results = {}
        for key, value in analysis_results.items():
            if isinstance(value, pd.DataFrame):
                serializable_results[key] = value.to_dict()
            elif isinstance(value, pd.Series):
                serializable_results[key] = value.to_dict()
            elif isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            else:
                serializable_results[key] = value

        # Include enhanced_results if OpenAI integration was performed
        if os.getenv('AIPROXY_TOKEN') and 'enhanced_results' in locals():
            for key, value in enhanced_results.items():
                if key not in serializable_results:
                    if isinstance(value, pd.DataFrame):
                        serializable_results[key] = value.to_dict()
                    elif isinstance(value, pd.Series):
                        serializable_results[key] = value.to_dict()
                    elif isinstance(value, np.ndarray):
                        serializable_results[key] = value.tolist()
                    else:
                        serializable_results[key] = value

        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        print(f"Analysis results saved to {results_path}\n")
    except Exception as e:
        print(f"Error saving analysis results: {e}\n")

    try:
        print("-----Generating a story------")
        generate_story(df, output_dir, enhanced_results)
        print("----finished generating story-----")
    except Exception as e:
        print(f"error generating story: {e}")

if __name__ == "__main__":
    main()




# # imports 
# import os
# import sys
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import httpx
# import chardet
# from pathlib import Path
# import argparse
# import requests
# import json
# import openai
# import warnings
# warnings.filterwarnings('ignore')
# import ast
# import re
# import time
# from typing import Dict, Any
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# from scipy import stats
# from scipy.cluster.hierarchy import linkage


# # Constants
# API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
# os.environ['AIPROXY_TOKEN'] = 'eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjEwMDI5NDBAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.Kymefw6W0esqnKNCPMT2njCfj6bNi7c-AsVhJD3R7n8'


# def do_generic_analysis(df, cluster_num=3, outlier_z=3):
#     """
#     Perform generic analysis on a pandas DataFrame including summary statistics,
#     missing values count, correlation matrix, outlier detection, clustering,
#     and hierarchical clustering.

#     Parameters:
#     - df: pandas.DataFrame
#         The input dataset.
#     - cluster_num: int, default=3
#         Number of clusters for K-Means.
#     - outlier_z: float, default=3
#         Z-score threshold for outlier detection.

#     Returns:
#     - analysis_results: dict
#         A dictionary containing various analysis results.
#     """
#     analysis_results = {}
    
#     # 1. Summary Statistics
#     summary = df.describe(include='all').transpose()
#     analysis_results['summary_statistics'] = summary
#     print("=== Summary Statistics ===")
#     print(summary)
#     print("\n" + "="*50 + "\n")
    
#     # 2. Missing Values
#     missing_values = df.isnull().sum()
#     missing_percent = (df.isnull().sum() / len(df)) * 100
#     missing_df = pd.DataFrame({
#         'Missing Values': missing_values,
#         'Percentage (%)': missing_percent
#     })
#     analysis_results['missing_values'] = missing_df
#     print("=== Missing Values ===")
#     print(missing_df)
#     print("\n" + "="*50 + "\n")
    
#     # 3. Correlation Matrix
#     numeric_df = df.select_dtypes(include=[np.number])
#     if not numeric_df.empty:
#         corr_matrix = numeric_df.corr()
#         analysis_results['correlation_matrix'] = corr_matrix
#         print("=== Correlation Matrix ===")
#         print(corr_matrix)
#     else:
#         print("No numerical columns available for correlation matrix.")
#     print("\n" + "="*50 + "\n")
    
#     # 4. Outlier Detection
#     outliers = {}
#     for col in numeric_df.columns:
#         z_scores = np.abs(stats.zscore(numeric_df[col].dropna()))
#         outlier_indices = numeric_df[col].dropna().index[z_scores > outlier_z].tolist()
#         outliers[col] = outlier_indices
#     analysis_results['outliers'] = outliers
#     print(f"=== Outliers Detected (Z-score > {outlier_z}) ===")
#     for col, indices in outliers.items():
#         print(f"- {col}: {len(indices)} outliers at indices {indices}")
#     print("\n" + "="*50 + "\n")
    
#     # 5. Clustering
#     # Preprocessing: Handle missing values and encode categorical variables
#     processed_df = df.copy()
    
#     # Fill numerical missing values with mean
#     for col in numeric_df.columns:
#         processed_df[col].fillna(processed_df[col].mean(), inplace=True)
    
#     # Encode categorical variables
#     categorical_cols = processed_df.select_dtypes(include=['object', 'category']).columns
#     le = LabelEncoder()
#     for col in categorical_cols:
#         processed_df[col] = processed_df[col].astype(str)
#         processed_df[col].fillna('Missing', inplace=True)
#         processed_df[col] = le.fit_transform(processed_df[col])
    
#     # Scaling
#     scaler = StandardScaler()
#     scaled_features = scaler.fit_transform(processed_df)
    
#     # K-Means Clustering
#     kmeans = KMeans(n_clusters=cluster_num, random_state=42)
#     clusters = kmeans.fit_predict(scaled_features)
#     processed_df['Cluster'] = clusters
#     analysis_results['kmeans_clusters'] = clusters
#     print(f"=== K-Means Clustering ===")
#     print(f"Applied K-Means clustering with {cluster_num} clusters.")
#     cluster_counts = pd.Series(clusters).value_counts().sort_index()
#     print("Cluster distribution:")
#     print(cluster_counts)
#     print("\n" + "="*50 + "\n")
    
#     # PCA for Visualization (Textual Summary)
#     pca = PCA(n_components=2)
#     principal_components = pca.fit_transform(scaled_features)
#     pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
#     pca_df['Cluster'] = clusters
#     explained_variance = pca.explained_variance_ratio_
#     analysis_results['pca_explained_variance'] = explained_variance
#     print("=== PCA for Clustering Visualization ===")
#     print(f"Explained variance by PC1: {explained_variance[0]:.2%}")
#     print(f"Explained variance by PC2: {explained_variance[1]:.2%}")
#     print("Principal Component 1 and 2 have been computed for clustering analysis.")
#     print("\n" + "="*50 + "\n")
    
#     # 6. Hierarchical Clustering
#     linked = linkage(scaled_features, method='ward')
#     analysis_results['hierarchical_linkage'] = linked
#     print("=== Hierarchical Clustering ===")
#     print("Performed hierarchical clustering using Ward's method.")
#     print("Linkage matrix has been computed for dendrogram analysis.")
#     print("\n" + "="*50 + "\n")
    
#     print("Generic analysis completed. All results are returned in text format.")
    
#     return analysis_results

# # Example usage:
# # if __name__ == "__main__":
# #     # Sample DataFrame for demonstration
# #     from sklearn.datasets import load_iris
# #     iris = load_iris()
# #     df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# #     df_iris['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    
# #     results = do_generic_analysis(df_iris, cluster_num=3, outlier_z=2.5)



# def analyze_with_openai(filename: str, df: pd.DataFrame, analysis_results: Dict[str, Any], additional_context: str = None) -> Dict[str, Any]:
#     """
#     Interact with OpenAI API to analyze the data based on generic analysis.
#     This function requests suggestions for additional analyses and Python code snippets
#     from the LLM, executes the code with caution, and provides enhanced insights.

#     Parameters:
#     - filename: str
#         The name of the dataset file.
#     - df: pandas.DataFrame
#         The input dataset.
#     - analysis_results: dict
#         The dictionary containing results from do_generic_analysis.
#     - additional_context: str, optional
#         Any additional context or information about the dataset.

#     Returns:
#     - enhanced_results: dict
#         A dictionary containing all analysis results, including those from OpenAI.
#     """
#     # Retrieve API key from environment variable
#     openai_api_key = os.getenv('AIPROXY_TOKEN')
#     if not openai_api_key:
#         print("Error: OpenAI API key not found in 'AIPROXY_TOKEN' environment variable.")
#         return {}
    
#     # Set OpenAI API configuration
#     api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

#     # Prepare the prompt with dataset details
#     column_info = "\n".join([f"- {col}: {df[col].dtype}" for col in df.columns])
#     summary_stats = analysis_results.get('summary_statistics', pd.DataFrame()).to_string()
#     missing_values = analysis_results.get('missing_values', pd.DataFrame()).to_string()
#     correlation_matrix = analysis_results.get('correlation_matrix', pd.DataFrame()).to_string()
    
#     # Outliers: summarize number of outliers per column
#     outliers = analysis_results.get('outliers', {})
#     outlier_summary = "\n".join([f"- {col}: {len(indices)} outliers" for col, indices in outliers.items()])
    
#     # Cluster distribution
#     clusters = analysis_results.get('kmeans_clusters', [])
#     if isinstance(clusters, pd.Series):
#         cluster_counts = clusters.value_counts().sort_index().to_dict()
#     else:
#         cluster_counts = pd.Series(clusters).value_counts().sort_index().to_dict()
    
#     # PCA explained variance
#     pca_variance = analysis_results.get('pca_explained_variance', [])
#     pca_info = ""
#     if len(pca_variance) >= 2:
#         pca_info = f"PC1: {pca_variance[0]:.2%}, PC2: {pca_variance[1]:.2%}"
#     elif len(pca_variance) == 1:
#         pca_info = f"PC1: {pca_variance[0]:.2%}"
#     else:
#         pca_info = "No PCA variance information available."
    
#     # Prepare prompt without requesting a summary
#     prompt = f"""
# You are an expert data analyst. Analyze the following dataset information and provide insights.

# **Filename**: {filename}

# **Columns and Types**:
# {column_info}

# **Summary Statistics**:
# {summary_stats}

# **Missing Values**:
# {missing_values}

# **Correlation Matrix**:
# {correlation_matrix}

# **Outliers**:
# {outlier_summary}

# **K-Means Clusters**:
# Cluster distribution: {cluster_counts}

# **PCA Explained Variance**:
# {pca_info}

# **Hierarchical Clustering**:
# Linkage matrix computed.

# **Additional Context**:
# {additional_context if additional_context else "None"}

# Based on the above information, please perform the following tasks:

# 1. **Suggest Additional Analyses**: Recommend specific analyses or function calls that could yield more insights into the data.

# 2. **Provide Python Code Snippets**: For each suggested analysis, provide Python code that can be executed to perform the analysis. Ensure that the code utilizes the existing `df` DataFrame and does not reload the dataset or use anyother variable that hasn't been defined previously.

# Ensure that the code is syntactically correct and handles potential errors gracefully.
# """

#     # Define a function to handle retries
#     def call_openai_api_with_retries(prompt, max_retries=3, backoff_factor=2):
#         headers = {
#             "Content-Type": "application/json",
#             "Authorization": f"Bearer {openai_api_key}"
#         }
#         data = {
#             "model": "gpt-4o-mini",  # Replace with "gpt-4" if incorrect
#             "messages": [
#                 {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI."},
#                 {"role": "user", "content": prompt}
#             ],
#             "max_tokens": 1500,
#             "temperature": 0.7
#         }

#         for attempt in range(1, max_retries + 1):
#             try:
#                 response = requests.post(api_url, headers=headers, data=json.dumps(data), timeout=60)
#                 if response.status_code == 200:
#                     return response.json()
#                 else:
#                     print(f"Attempt {attempt}: Received status code {response.status_code} with message: {response.text}")
#             except requests.exceptions.RequestException as e:
#                 print(f"Attempt {attempt} failed with error: {e}")

#             if attempt < max_retries:
#                 sleep_time = backoff_factor ** attempt
#                 print(f"Retrying in {sleep_time} seconds...")
#                 time.sleep(sleep_time)
#             else:
#                 print("All retry attempts failed.")
#                 return None

#     # Call the API with retries
#     response_json = call_openai_api_with_retries(prompt)
#     if not response_json:
#         print("Failed to get a response from OpenAI API.")
#         return analysis_results  # Return existing analysis results

#     # Extract the answer from the response
#     try:
#         answer = response_json['choices'][0]['message']['content'].strip()
#     except (KeyError, IndexError) as e:
#         print(f"Error parsing response JSON: {e}")
#         print("Full response:", response_json)
#         return analysis_results  # Return existing analysis results

#     print("\n=== OpenAI Analysis ===\n")
#     print(answer)
#     print("\n" + "="*50 + "\n")

#     # Initialize enhanced_results with existing analysis_results
#     enhanced_results = analysis_results.copy()

#     # Parse the answer to extract different sections
#     # Assuming the LLM formats sections with headers like ### Additional Analyses
#     suggestions_match = re.search(r'###\s*Additional Analyses\s*([\s\S]*?)(?:\n###|$)', answer, re.IGNORECASE)
#     code_snippets = re.findall(r'```python\n(.*?)```', answer, re.DOTALL)

#     if suggestions_match:
#         suggestions_text = suggestions_match.group(1).strip()
#         # Assume suggestions are in a list format
#         suggestions = re.findall(r'\d+\.\s*(.+)', suggestions_text)
#         enhanced_results['openai_suggestions'] = suggestions
#         print("=== OpenAI Suggestions for Additional Analyses ===")
#         for idx, suggestion in enumerate(suggestions, 1):
#             print(f"{idx}. {suggestion}")
#         print("\n" + "="*50 + "\n")
#     else:
#         # Fallback if specific sections are not found
#         enhanced_results['openai_suggestions'] = []
#         print("No specific suggestions found from OpenAI.\n" + "="*50 + "\n")

#     # Execute each code snippet and capture results
#     for idx, code in enumerate(code_snippets, 1):
#         print(f"--- Executing Code Snippet {idx} ---")
#         try:
#             # Define a local namespace for code execution
#             local_namespace = {
#                 'df': df,
#                 'analysis_results': enhanced_results
#             }
#             exec(code, globals(), local_namespace)
#             print(f"Code Snippet {idx} executed successfully.\n")
#             # Retrieve any updates made to 'analysis_results' or 'enhanced_results'
#             if 'analysis_results' in local_namespace:
#                 for key, value in local_namespace['analysis_results'].items():
#                     enhanced_results[key] = value
#             if 'enhanced_results' in local_namespace:
#                 for key, value in local_namespace['enhanced_results'].items():
#                     enhanced_results[key] = value
#         except Exception as e:
#             print(f"Error executing Code Snippet {idx}: {e}\n")
#     print("OpenAI analysis and code execution completed.\n" + "="*50 + "\n")

#     return enhanced_results



# if __name__ == "__main__":
#     from sklearn.datasets import load_iris
#     # "C:\Users\umang\OneDrive\Desktop\TDS\TDS Project 2\goodreads.csv"
#     # Load and prepare the Iris dataset
#     # iris = load_iris()
#     # df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
#     # df_iris['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    
#     # # Perform generic analysis
#     # results = do_generic_analysis(df_iris, cluster_num=3, outlier_z=2.5)
    
#     # # Analyze with OpenAI
#     # enhanced_results = analyze_with_openai(
#     #     filename="iris_dataset.csv",
#     #     df=df_iris,
#     #     analysis_results=results,
#     #     additional_context="This is the Iris dataset containing measurements of iris flowers."
#     # )
    
#     df_gg = pd.read_csv('goodreads.csv')
#     results = do_generic_analysis(df_gg, cluster_num=3, outlier_z=2.5)
#     enhanced_results = analyze_with_openai(
#         filename="goodreads.csv",
#         df=df_gg,
#         analysis_results=results
#     )

#     # Display enhanced results
#     print("=== Enhanced Analysis Results ===")
#     for key, value in enhanced_results.items():
#         print(f"{key}:")
#         if isinstance(value, pd.DataFrame):
#             print(value.head())
#         elif isinstance(value, pd.Series):
#             print(value.to_dict())
#         elif isinstance(value, list):
#             for item in value:
#                 print(f"- {item}")
#         else:
#             print(value)
#         print("\n" + "-"*50 + "\n")
