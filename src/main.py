###################################
# Main Runner for Application
###################################
import subprocess
import os

# Main function
if __name__ == '__main__':
    # ensure files directory exists at root of project
    if not os.path.exists('./files'):
        os.makedirs('./files')

    # Run the experiments    
    subprocess.run(["python", "part1.py"])
    subprocess.run(["python", "part2.py"])