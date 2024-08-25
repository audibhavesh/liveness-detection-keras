import subprocess

if __name__ == '__main__':
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
