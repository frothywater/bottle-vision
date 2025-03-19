import os
import time
from argparse import ArgumentParser
from typing import Optional

import boto3


def download_dataset(local_dir: str, dataset_dir: str, max_tar_index: Optional[int]):
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

    # List all objects in the bucket
    keys = []
    dirs = []
    next_token = ""
    base_kwargs = {
        "Bucket": s3_bucket,
        "Prefix": dataset_dir,
    }
    while next_token is not None:
        kwargs = base_kwargs.copy()
        if next_token != "":
            kwargs.update({"ContinuationToken": next_token})
        results = s3.list_objects_v2(**kwargs)
        contents = results.get("Contents")
        for i in contents:
            k = i.get("Key")
            if k[-1] != "/":
                keys.append(k)
            else:
                dirs.append(k)
        next_token = results.get("NextContinuationToken")

    # Download all objects in the bucket
    for d in dirs:
        dest_pathname = os.path.join(local_dir, d)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
    for i, k in enumerate(keys):
        # check if the tar file index is within the range
        basename = os.path.basename(k)
        stem, ext = os.path.splitext(basename)
        if ext == ".tar":
            tar_index = int(stem.split("-")[-1])
            if max_tar_index is not None and tar_index > max_tar_index:
                print(f"{k} is beyond the maximum tar index, skipping")
                continue

        dest_pathname = os.path.join(local_dir, k)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))

        if os.path.exists(dest_pathname) and os.path.getsize(dest_pathname) > 0:
            print(f"[{i + 1}/{len(keys)}] {k} already exists, skipping")
            continue

        start = time.time()
        s3.download_file(s3_bucket, k, dest_pathname)
        elapsed = time.time() - start
        speed_mbps = os.path.getsize(dest_pathname) / elapsed / 1024 / 1024

        print(f"[{i + 1}/{len(keys)}] Downloaded {k} to {dest_pathname}, {speed_mbps:.2f} MB/s")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("local_dir", type=str, help="Local directory to download the dataset")
    parser.add_argument("dataset_dir", type=str, help="Directory in the S3 bucket to download")
    parser.add_argument("--max_tar_index", type=int, help="Maximum index of tar files to download")
    args = parser.parse_args()

    download_dataset(args.local_dir, args.dataset_dir, args.max_tar_index)
