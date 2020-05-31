import os


def files_number(path):
    files_counted = 0

    folders = ([name for name in os.listdir(path)])

    for folder in folders:
        contents = os.listdir(os.path.join(path, folder))
        files_counted += len(contents)

    return files_counted

dir_path = os.getcwd()

training_path = dir_path + '/datasets/training'
print("Number of data training: ", files_number(training_path))

validation_path = dir_path + '/datasets/validation'
print("Number of data validation: ", files_number(validation_path))
