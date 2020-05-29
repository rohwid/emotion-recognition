import os


def files_number(path):
    files_counted = 0

    folders = ([name for name in os.listdir(path)])

    for folder in folders:
        contents = os.listdir(os.path.join(path, folder))
        files_counted += len(contents)

    return files_counted


training_path = '/home/rohwid/Documents/PyCharm/emotion-recognition/datasets/training'
print("Number of data training: ", files_number(training_path))

validation_path = '/home/rohwid/Documents/PyCharm/emotion-recognition/datasets/validation'
print("Number of data validation: ", files_number(validation_path))
