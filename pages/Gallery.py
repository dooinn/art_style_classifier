import streamlit as st
import os
import PIL.Image as Image
from io import BytesIO

def load_images_from_folder(folder):
    """Loads images from a specified folder and returns a list of images."""
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, filename)
            try:
                with open(img_path, "rb") as file:
                    img = Image.open(file)
                    img = img.resize((150, 150))  # Resize to make all images the same size
                    images.append(img)
            except Exception as e:
                print(e)
    return images

def display_images(image_list, rows_per_page, cols_per_page):
    """Display a grid of images with pagination."""
    total_images = len(image_list)
    images_per_page = rows_per_page * cols_per_page
    pages = total_images // images_per_page + (total_images % images_per_page > 0)
    page = st.slider("Choose a page", 1, pages)
    start = (page - 1) * images_per_page
    end = start + images_per_page
    images_to_show = image_list[start:end]

    index = 0
    for row in range(rows_per_page):
        cols = st.columns(cols_per_page)
        for col in cols:
            if index < len(images_to_show):
                col.image(images_to_show[index], use_column_width=True)
                index += 1

# Directory setup
base_dir = "notebook/dataset/test"
categories = os.listdir(base_dir)
selected_category = st.sidebar.selectbox("Choose a category", categories)

# Load images
image_folder = os.path.join(base_dir, selected_category)
images = load_images_from_folder(image_folder)

st.title("Gallery")

# Display images with pagination
display_images(images, rows_per_page=6, cols_per_page=4)  # 6 rows, 4 columns per page
