FROM public.ecr.aws/lambda/python:3.12

# Install Alpine Linux packages needed for LightGBM
RUN apk add --no-cache libgomp

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY models ./models

CMD ["app.main.handler"]
