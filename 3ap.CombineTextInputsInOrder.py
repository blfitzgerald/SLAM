import os
import glob
import sys

def combine_text_files(input_folder, output_file_name, pattern):
    """
    Combines text files in the input_folder matching the given pattern into a single file.

    Parameters:
    - input_folder (str): Path to the folder containing text files.
    - output_file_name (str): Name of the output file (default is "combined_output.txt").
    - pattern (str): Pattern to match file names (default is "Input.hr.*.txt").
    """
    # Create the full path pattern for matching files
    file_pattern = os.path.join(input_folder, pattern)
    
    # Find all files matching the pattern and sort them by the numerical part
    files_to_combine = sorted(glob.glob(file_pattern), key=lambda x: int(x.split('.')[-2]))
    
    if not files_to_combine:
        print(f"No files found matching the pattern '{pattern}' in {input_folder}.")
        return
    
    # Construct the full path for the output file in the input_folder
    output_file = os.path.join(input_folder, output_file_name)
    
    # Combine files
    combined_data = []
    for file_path in files_to_combine:
        print(f"Processing file: {file_path}")
        with open(file_path, 'r') as file:
            combined_data.append(file.read())
    
    # Write combined data to the output file
    with open(output_file, 'w') as output:
    	output.write("".join(combined_data))

    
    print(f"Combined {len(files_to_combine)} files into {output_file}")

dur= sys.argv[2]

input_folder = sys.argv[1]  # Replace with your folder path
output_file_name="Inputs."+dur+"hr.txt" 
pattern="Input."+dur+"hr.*.txt"


combine_text_files(input_folder, output_file_name, pattern)
