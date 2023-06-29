import os
import imageio

# Directory containing the PNG files
directory = 'images/'

# Create a list of file names in the directory
file_names = sorted(os.listdir(directory))

# Initialize the video writer
output_file = 'Hx_video.mp4'
writer = imageio.get_writer(output_file, fps=30)  # Adjust the frame rate (fps) as needed

# Loop through the file names and add frames to the video
for file_name in file_names:
    # Check if the file is a PNG file
    if file_name.endswith('.png'):
        file_path = os.path.join(directory, file_name)

        # Read the PNG file and add it as a frame in the video
        image = imageio.imread(file_path)
        writer.append_data(image)

# Close the video writer
writer.close()

print(f'Video saved: {output_file}')
