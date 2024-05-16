# ArtStyle & Artists WebApp

<div style="position: relative; padding-bottom: 56.25%; height: 0;"><iframe src="https://www.loom.com/embed/c1b966b9bfbe4c339d2be6c5c66e70e0?sid=90626517-7fc3-4dc0-9426-60aaab4d944c" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>



Welcome to the ArtStyle & Artists WebApp! This project showcases my skills in PyTorch and NLP by leveraging machine learning and data science techniques to identify art styles from images and provide information about artists along with recommendations of similar artists.

## Overview

The web app is divided into two main sections:

1. **ArtStyle**: Upload an image of a painting, and the app identifies the artistic style. Detailed information about the identified art style is provided.
2. **Artists**: Get information about various artists, including their biographies and images, and receive recommendations of similar artists.

## Technologies Used

- **Streamlit**: For building and sharing data applications.
- **PyTorch**: For deep learning model implementation.
- **timm**: For utilizing pretrained image models.
- **Pandas**: For data manipulation.
- **Requests**: For making HTTP requests.
- **scikit-learn**: For machine learning tasks.

## Machine Learning Algorithms

### ArtStyle Section

1. **Model Architecture**:
    - Uses a custom neural network based on the EfficientNet-B0 architecture.
    - Classifies images into one of 12 art styles: Symbolism, Surrealism, Supermatism, Romanticism, Renaissance, Primitivism, Post-Impressionism, Pop Art, Impressionism, Expressionism, Cubism, and Baroque.

2. **Training**:
    - The base model is pretrained on ImageNet.
    - A new classification head is added and fine-tuned for art style classification.
    - The model is trained using the Adam optimizer and CrossEntropyLoss criterion.

3. **Inference**:
    - Preprocesses uploaded images and predicts the art style using the trained model.
    - Fetches information about the predicted art style from a CSV file.

### Artists Section

1. **Similarity Recommendation**:
    - Uses a cosine similarity matrix to recommend similar artists based on a selected artist.
    - The similarity matrix is computed using a CountVectorizer on artist tags.

2. **Fetching Artist Information**:
    - Fetches information from a CSV file.
    - Additional details are retrieved from Wikipedia using the Wikipedia API.

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/artstyle-artists-webapp.git
    cd artstyle-artists-webapp
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Project Structure

- `app.py`: The main Streamlit app file.
- `data/`: Contains CSV files with art style and artist information.
- `artifacts/`: Contains the trained model and similarity matrix.
- `assets/`: Contains image assets used in the app.
- `dataset/`: Contains training, validation, and test datasets.

## Future Improvements
- Enhance the accuracy of the art style classification model (currently experiencing classification issues).
- Expand the model to recognize more art styles.
- Enhance the similarity recommendation system with additional features.
- Add more detailed artist information and multimedia content.

## Image Data Source

The dataset used for training the art style classification model is sourced from Kaggle:

- **Best Artworks of All Time**: [Kaggle Dataset](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time)
Explore the world of art with the ArtStyle & Artists WebApp! Feel free to contribute or provide feedback.
