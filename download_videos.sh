echo ">>> Adding package gdown"
uv add gdown
#Or use pip if you dont use uv
#pip install gdown
echo ">>> Downloading videos from my google drive"
gdown --folder https://drive.google.com/drive/u/0/folders/1MQ2loTtn1RTHMlPMRSCHvCBE71CvzDq8\
     --continue \
     -O ./input_videos