import subprocess

import pickle


def train():
    print("")

    # Define the command as a list of strings
    command = [
        'python',
        'train_liveness.py',
        '--dataset', 'dataset',
        '--model', 'liveness.keras',
        '--le', 'le.pickle'
    ]

    # Run the command
    subprocess.run(command)


def run():  # Define the command and its arguments
    command = [
        'python3', 'liveness_demo.py',
        '--model', 'liveness.keras',
        '--le', 'le.pickle',
        '--detector', 'face_detector'
    ]

    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True)

    # Print the output and errors
    print("Output:\n", result.stdout)
    print("Errors:\n", result.stderr)


def tflite_run():  # Define the command and its arguments
    command = [
        'python', 'tflite_demo.py',
        '--model', 'liveness_model.tflite',
        '--le', 'le.pickle',
        '--detector', 'face_detector'
    ]

    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True)

    # Print the output and errors
    print("Output:\n", result.stdout)
    print("Errors:\n", result.stderr)


def create_dataset():  # Define the command and its arguments
    command = [
        'python', 'gather_examples.py',
        '-i', 'videos/real3.mp4',
        '-o', 'dataset/real3',
        '--detector', 'face_detector',
        "--skip", "4"
    ]

    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True)

    # Print the output and errors
    print("Output:\n", result.stdout)
    print("Errors:\n", result.stderr)


if __name__ == '__main__':
    # train()
    # run()
    # with open("le.pickle", "rb") as f:
    #     le = pickle.load(f)
    #
    # print("CLASSS ")
    # print(le.classes_)
    tflite_run()
    # create_dataset()
