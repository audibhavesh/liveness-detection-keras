import subprocess


def train():
    print("")

    # Define the command as a list of strings
    command = [
        'python3',
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
        'python3', 'tflite_demo.py',
        '--model', 'liveness_model.tflite',
        '--le', 'le.pickle',
        '--detector', 'face_detector'
    ]

    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True)

    # Print the output and errors
    print("Output:\n", result.stdout)
    print("Errors:\n", result.stderr)


if __name__ == '__main__':
    # train()
    # run()
    tflite_run()
