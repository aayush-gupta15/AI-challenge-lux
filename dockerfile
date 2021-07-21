FROM python:3.7


RUN pip3 install --upgrade pip

# Install the python requirements from requirements.txt
COPY requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt
# Replace Pillow with Pillow-SIMD to take advantage of AVX2
RUN pip3 uninstall -y pillow && CC="cc -mavx2" pip3 install -U --force-reinstall pillow-simd

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python

COPY . ./

# Set the CMD to your handler
CMD [ "python","-u", "./notifier.py" ]