import os
from argparse import ArgumentParser

import boto3


def download_dataset(local_dir: str, dataset_dir: str):
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
    for k in keys:
        dest_pathname = os.path.join(local_dir, k)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
        s3.download_file(s3_bucket, k, dest_pathname)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("local_dir", type=str, help="Local directory to download the dataset")
    parser.add_argument("dataset_dir", type=str, help="Directory in the S3 bucket to download")
    args = parser.parse_args()

    download_dataset(args.local_dir, args.dataset_dir)
