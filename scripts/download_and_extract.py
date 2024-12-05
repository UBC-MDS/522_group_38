import click
import requests
import zipfile
import os

@click.command()
@click.option('--url', type=str, help='URL to download the data from', required=True)
@click.option('--output_dir', type=str, help='Directory to save and extract the data', required=True)
def download_and_extract(url, output_dir):
    """
    Downloads and extracts the data from a given URL.

    Args:
    - url (str): The URL of the dataset.
    - output_dir (str): The directory to save and extract the dataset.
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Download the file
        print(f"Downloading data from {url}...")
        response = requests.get(url)
        zip_path = os.path.join(output_dir, "wine_quality.zip")
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        print(f"Data downloaded to {zip_path}")

        # Extract the zip file
        print(f"Extracting data to {output_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print("Data extraction complete.")
    except Exception as e:
        print(f"Error during data download or extraction: {e}")

if __name__ == '__main__':
    download_and_extract()
