import os

# Input file path
input_file_path = '/media/hamed/Data/CV_PRJ/kitti_tracking/data_tracking_label_2/training/label_02/0007.txt'

# Output directory to store separate files
output_directory = '/media/hamed/Data/CV_PRJ/kitti_tracking/data_tracking_label_2/training/label_02/tracking_plus_detection/'
os.makedirs(output_directory, exist_ok=True)

# Open and read the input file
with open(input_file_path, 'r') as input_file:
    current_frame = 0  # Initialize to a starting frame number
    current_frame_lines = []

    # Iterate through each line in the input file
    for line in input_file:
        # Split the line into columns
        columns = line.strip().split()

        # Extract frame number
        frame = int(columns[0])

        # Check if a new frame has started
        if frame != current_frame:
            # If yes, create an empty file for missing frames
            for missing_frame in range(current_frame + 1, frame):
                missing_frame_file_path = os.path.join(output_directory, f'{missing_frame:06d}.txt')
                with open(missing_frame_file_path, 'w'):
                    pass  # Create an empty file

            # Write the previous frame information to a separate file
            if current_frame is not None:
                output_file_path = os.path.join(output_directory, f'{current_frame:06d}.txt')
                with open(output_file_path, 'w') as output_file:
                    output_lines = [' '.join([f'{float(value):.2f}' if value[1:].replace('.', '', 1).isdigit() else value for value in row]) for row in current_frame_lines]
                    output_file.write('\n'.join(output_lines))

            # Reset for the new frame
            current_frame = frame
            current_frame_lines = []

        # Append the current line to the lines for the current frame
        current_frame_lines.append(columns[2:])

    # Write the last frame information to a separate file
    if current_frame is not None:
        output_file_path = os.path.join(output_directory, f'{current_frame:06d}.txt')
        with open(output_file_path, 'w') as output_file:
            output_lines = [' '.join([f'{float(value):.2f}' if value[1:].replace('.', '', 1).isdigit() else value for value in row]) for row in current_frame_lines]
            output_file.write('\n'.join(output_lines))
