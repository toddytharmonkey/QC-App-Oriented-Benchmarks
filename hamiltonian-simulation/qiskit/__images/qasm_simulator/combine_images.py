from PIL import Image, ImageDraw, ImageFont


def save_and_combine_images(images, output_filename, method):
    # Assuming all images are the same size
    width, height = images[0].size

    # Create a new image to accommodate 2x2 grid with extra space for text
    combined_image = Image.new('RGB', (2 * width, 2 * height + 100), 'white')  # Adjusted space for labels and title

    # Create drawing object
    draw = ImageDraw.Draw(combined_image)

    # Use a larger font size; download a .ttf file or use available system fonts
    try:
        font = ImageFont.truetype("arial.ttf", size=24)  # Specify path to a TTF font file and size
    except IOError:
        font = ImageFont.load_default()

    # Set text for title and labels
    title = f"Hamiltonian Simulation Method {method}"
    y_label = "2Q gate fidelity"
    x_labels = ["No compilation", "Pytket gate compilation"]

    # Calculate center for the title and add it

    # Paste the images and add axis labels
    for index, image in enumerate(images):
        x = (index % 2) * width
        y = (index // 2) * height + 50  # Adjust for title space

        # Paste image
        combined_image.paste(image, (x, y))

        # Center x-axis labels below each image

    # Y-axis labels (if more precision needed, adjust the positions)

    # Save the new image
    combined_image.save(output_filename)
if __name__ == "__main__":

    combined_image_suffix = ""

    for method in [1]:

        images = []

        for f in (.95, .995):
            for use_pytket in [False, True]: 

                file_name = f"Hamiltonian-Simulation-vplot{method}_{f}_{use_pytket}" + combined_image_suffix +  ".jpg" 

                images.append(Image.open(file_name))


        save_and_combine_images(images, f"combined_vplots_method_{method}" + combined_image_suffix + ".jpg", method)


