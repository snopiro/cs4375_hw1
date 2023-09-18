# CS4375 Assignment 1
GitHub Repo: https://github.com/snopiro/cs4375_hw1

## Project Description
This is a an assignment for my Machine Learning class. The goal is to take a dataset and develop a linear regression model to predict a variable. This code takes a publicly available dataset on cars, like their cylinders, horsepower, acceleration, mpg, etc, and tries to predict mpg based on the other factors of the car. The program utilizes gradient descent on multiple variables to develop the model.

## How to Run
There are several ways you can run this project. I have provided docker files to provide an execution environment if you have Docker installed. Alternatively, you can also set up a Github Codespaces environment and run the docker files from there. You can also run the project locally if you have python installed on your machine.

After running the project, it will create log files and plots in the src/files directory.

### Docker
If you would like to use Docker on your local machine, install Docker desktop and run the following command in the root directory of this project, where the docker-compose.yml file is located:
```
docker compose up
```

### Github Codespaces (Recommended)

If you would like to use Github Codespaces, click the green button at the top of this repository that says "Code" and select "Create codespace on master". This will open a new Codespaces environment with the project already cloned. Once the environment is ready, navigate to the root directory of this project, where the docker-compose.yml file is located and run the following command in the terminal:
```
docker compose up
```

### Running the project locally

If you have python already installed on your machine, as well as the libraries
```
numpy
pandas
matplotlib
scikit-learn
```
you can run the project locally by navigating to the src directory of this project and running the following command:
```
python main.py
```
