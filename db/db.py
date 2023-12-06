from minio import Minio

client = Minio(
                f"127.0.0.1:9000",
                access_key='2aMFSfJf0Ar6bTjlLj48',
                secret_key='bI3l3Thl2PLEA9eRcZeLjnHlePq2iwVw47l1p3iP',
                secure=False
                )

buckets = client.list_buckets()
for bucket in buckets:
    print(bucket.name, bucket.creation_date)