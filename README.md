# Face Image Merging Project

Welcome to the **Face Image Merging Project** repository! This project demonstrates a quick and fun experiment where two images of people's faces are merged into one using the Face-SPARNet repository. This repository contains the necessary files and scripts to perform this task.

## Project Overview

- **Image Merging**: This project utilizes the Face-SPARNet repository to merge two separate images of people's faces into a single composite image. The resulting image combines facial features from both input images to create an intriguing visual effect.

## Repository Structure

Here's an overview of the files within this repository:

- `mergefaces.py`: This Python script is the heart of the project. It uses the Face-SPARNet model to merge two input images of faces. You can customize the input images by providing their file paths within the script.

- `.gitignore`: A standard Git ignore file to exclude unnecessary files and directories from version control.

- `requirements.txt`: Lists the required Python packages and their versions for this project. You can install these dependencies using `pip install -r requirements.txt`.

- `shape_predictor_68_face_landmarks.dat`: This file contains facial landmark information necessary for the Face-SPARNet model to perform the image merging.

## How to Use

To merge two images of people's faces using this project, follow these steps:

1. Clone this repository to your local machine.

2. Install the required dependencies by running the following command:

pip install -r requirements.txt

3. Prepare two images of people's faces that you want to merge. Make sure they are in a compatible format, such as JPEG or PNG.

4. Open the `mergefaces.py` script and update the file paths of the input images in the setOfNames list.

5. Run the `mergefaces.py` script using Python. This will merge the input images and display them.

6. Check the `output.png` file to see the merged image of the two faces.

7. Experiment with different input images to create unique face merges!

## Credits

This project is a quick experiment made possible by the Face-SPARNet repository. Special thanks to the developers and contributors of the Face-SPARNet project for providing the tools to create fascinating face merges.

If you have any questions or suggestions, please feel free to reach out.

Anish Karthik
