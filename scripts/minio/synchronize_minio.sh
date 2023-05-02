# ! /bin/bash

rm -fR dataset
mkdir dataset
aws --profile minio --endpoint-url http://192.168.1.194:9000 s3 cp  s3://dataset dataset --recursive

rm -fR data
mkdir data
aws --profile minio --endpoint-url http://192.168.1.194:9000 s3 cp  s3://data data --recursive

rm -fR config
mkdir config
aws --profile minio --endpoint-url http://192.168.1.194:9000 s3 cp  s3://config config --recursive