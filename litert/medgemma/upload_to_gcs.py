import os
import glob
from google.cloud import storage

def upload_models(bucket_name, destination_folder, patterns=["*.tflite", "*.task"]):
    """Uploads all files matching patterns to a GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    files_to_upload = []
    for pattern in patterns:
        files_to_upload.extend(glob.glob(pattern))

    if not files_to_upload:
        print("No files matching patterns found.")
        return

    print(f"Found {len(files_to_upload)} files to upload.")

    for local_file in files_to_upload:
        blob_name = os.path.join(destination_folder, os.path.basename(local_file))
        blob = bucket.blob(blob_name)
        
        print(f"Uploading {local_file} to gs://{bucket_name}/{blob_name}...")
        blob.upload_from_filename(local_file)
        print(f"Uploaded {local_file} successfully.")

if __name__ == "__main__":
    BUCKET_NAME = "medgemma-litert"
    DESTINATION_FOLDER = "cpu-highmem8"
    
    upload_models(BUCKET_NAME, DESTINATION_FOLDER)
