## Download Datasets

Use this [link](https://www.dropbox.com/sh/un70d4x3exu9vg7/AAAlJrE7uj90BxN3a7_Js41va?dl=0) to download the datasets.

## Datasets Content

Here's the datasets content:
- `fer2013_7_emotions.7z`
- `fer2013_5_emotions.7z`
- `fer2013_csv_7_emotions.7z`

Extract and open! You will be find a folder named `datasets`. And here's the things thats you should deal with the folder.

### 1. `fer2013_7_emotions`

Move the folder named `datasets` to this specific project folder:

- `7_emotions_mobilenet`
- `7_emotions_vgg`

### 2. `fer2013_5_emotions`

Move the folder named `datasets` to this specific project folder:

- `5_emotions_mobilenet`
- `3_emotions_vgg`

### 3. `fer2013_csv_7_emotions`

Move the folder named `datasets` to this specific project folder:

- `7_emotions_cnn`

## How to Run the project

In each project folder there's a file named begin with `emotion` or `emotion_bla.. bla.py`. That's the file to does the training process with `datasets` with you've placed in there before.

    $ python emotion_bla bla.py

And the file named begin with `stream` or `stream_bla.. bla.py`. That's the file to does the realtime testing with the hardcascade file and the `*.h5` or `*.json` as the result of the training process.

    $ pyhton stream_bla.. bla.py
    
Hopefully you know what I mean :)