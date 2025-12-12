FROM amazonlinux:2023

# 1. Install system + Python 3.11
RUN dnf install -y \
    python3.11 \
    python3.11-pip \
    libgomp \
    gcc \
    gcc-c++ \
    make \
    unzip \
 && dnf clean all

# 2. Install Lambda Runtime Interface Client
RUN python3.11 -m pip install --no-cache-dir awslambdaric

# 3. Install Python dependencies
COPY requirements.txt /var/task/
RUN python3.11 -m pip install --no-cache-dir -r /var/task/requirements.txt

# 4. Copy app + models
COPY app /var/task/app
COPY models /var/task/models

# 5. Copy Lambda RIE
COPY bin/aws-lambda-rie /usr/local/bin/aws-lambda-rie
RUN chmod +x /usr/local/bin/aws-lambda-rie

WORKDIR /var/task

# 6. Lambda entrypoint
ENTRYPOINT ["/usr/local/bin/aws-lambda-rie", "/usr/bin/python3.11", "-m", "awslambdaric"]
CMD ["app.main.handler"]
