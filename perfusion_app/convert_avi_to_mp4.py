import os
import subprocess

input_dir = '/Volumes/SSD Konrad /nagrania AVI WIM'
output_dir = '/Volumes/SSD Konrad /WIM_mp4_files_BM'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(input_dir):
    if filename.endswith('.avi'):
        input_path = os.path.join(input_dir, filename)
        print(f"Converting {input_path}...")
        output_filename = os.path.splitext(filename)[0] + '.mp4'
        output_path = os.path.join(output_dir, output_filename)
        
        command = ['ffmpeg', '-i', input_path, output_path]
        subprocess.run(command)

print("Conversion completed.")