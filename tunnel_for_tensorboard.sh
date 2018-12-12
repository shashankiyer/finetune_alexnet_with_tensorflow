#!/bin/bash

ssh -L 127.0.0.1:6006:127.0.0.1:6006 -i AWS_data/Learning_to_hash.pem ubuntu@ec2-34-228-255-40.compute-1.amazonaws.com
