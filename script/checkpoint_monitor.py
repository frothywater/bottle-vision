import os
import time
from argparse import ArgumentParser

import boto3
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer


class CheckpointHandler(PatternMatchingEventHandler):
    def __init__(self, s3_client, bucket, **kwargs):
        super().__init__(patterns=["*.ckpt"], **kwargs)
        self.s3_client = s3_client
        self.bucket = bucket

    def on_created(self, event):
        self.upload_file(event.src_path)

    def on_modified(self, event):
        self.upload_file(event.src_path)

    def upload_file(self, filepath):
        filename = os.path.basename(filepath)
        print(f"Detected checkpoint: {filename}. Uploading to S3...")
        self.s3_client.upload_file(filepath, self.bucket, f"checkpoints/{filename}")
        print(f"Uploaded {filename} to S3 successfully.")


def monitor_checkpoints(path_to_monitor):
    # Retrieve S3 credentials and bucket info from environment variables
    s3_access_key = os.environ.get("S3_ACCESS_KEY")
    s3_secret_key = os.environ.get("S3_SECRET_KEY")
    s3_bucket = os.environ.get("S3_BUCKET")
    s3_endpoint = os.environ.get("S3_ENDPOINT", None)

    # Create a boto3 S3 client session
    session = boto3.session.Session()
    s3 = session.client(
        service_name="s3",
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
        endpoint_url=s3_endpoint,
    )

    # Start monitoring the directory
    event_handler = CheckpointHandler(s3, s3_bucket)
    observer = Observer()
    observer.schedule(event_handler, path=path_to_monitor, recursive=True)
    observer.start()
    print(f"Started monitoring {path_to_monitor} for .ckpt files.")

    # Keep the script running until interrupted
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path_to_monitor", type=str, help="Path to monitor for checkpoint files")
    args = parser.parse_args()

    # The path to monitor is the exp directory inside the cloned repo.
    monitor_checkpoints(args.path_to_monitor)
