# Nepali Handwritten Character Classification Using CNN

## Project Description:
This project aims to develop a deep learning model to classify handwritten Nepali characters into **46** different classes using Convolutional Neural Networks (CNNs). The dataset contains **92,000** grayscale images, each representing one of the Nepali characters **(क, ख, ग, ... + १, २, ...)**. The goal is to develop and train a deep learning model that can accurately classify these characters

- **Dataset Total Images:** 92,000 
- **Training Images:** 78,200 
- **Test Images:** 13,800 
- **Image Size:** 32x32 pixels 
- **Number of Classes:** 46 (includes both characters and numbers)

## Approach: 
### Model:
The model is a CNN with three convolutional layers followed by max-pooling and fully connected layers for classification. 
### Training: 
The model is trained using the training set and evaluated on the test set using accuracy as the primary metric. 
### Visualization: 
The project includes visualization scripts to display model predictions on a grid of test images.

## Tech Stack Used: 
- **PyTorch:** For building, training, and evaluating the CNN model. 
- **NumPy:** For loading and manipulating the dataset.
- **Matplotlib:** For visualizing the predictions. 
- **torchvision:** For dataset transformation.

## File Structure
- │── main.ipynb        # Main Jupyter Notebook containing the whole code
- │── dataset/          # Directory containing the dataset 
- │── requirements.txt  # List of dependencies required for the project
- │── README.md         # Project documentation (this file)

## Execution
### Installing Dependencies
 Make sure you have Python installed. Then, install the required dependencies using the requirements.txt file :
 
```bash
pip install -r requirements.txt
```

### How To Run
- Run the main.ipynb File.
This code is specifically designed to be run in a Jupyter Notebook (.ipynb format). Therefore, it is essential that you run the project using the Jupyter Notebook or any environment that can run .ipynb files.

- **To run the project:** Open the main.ipynb. Execute each cell step-by-step in the order they appear.

### Running on Google Colab:
If you prefer running on Google Colab, don't forget to mount the google drive and change the dataset path to match your Google Drive path.


## Model Training and Results

- **Epochs**: The model was trained across 5 epochs. An epoch represents one complete cycle through the entire training dataset, which helps the model to improve its learning.
- **Training Accuracy**: At the end of the 5th epoch, the model achieved a training accuracy of **97.60%**.
- **Test Accuracy**: The final test accuracy was **98.49%**, indicating the model's effectiveness on data it hasn't previously seen.
- **Loss**: The loss at the conclusion of the training was **0.0748** indicating that the model's predictions are closely aligned with the actual outcomes.
- **Total Images Trained**: Over the 5 epochs, the model processed a total of **391,000 images**.

## Conclusion
The model successfully classifies Nepali handwritten characters with a high accuracy of **98.49%** on the test data after training for 5 epochs. The results show that the model is effective at generalizing to unseen data, making it a reliable solution for recognizing Nepali characters.


## Authors: 
- ### Anoma Tuladhar
- ### Nairiti Rai
- ### Pradeep Bhatta 
- ### Samikshya Dahal
