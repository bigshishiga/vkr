import os
import argparse
from PIL import Image
from shutil import copytree
from tqdm import tqdm
import multiprocessing
from functools import partial

def concatenate_images_horizontally(img1_path, img2_path, output_path):
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    # Ensure both images have the same height
    if img1.height != img2.height:
        raise ValueError(f"Images {img1_path} and {img2_path} do not have the same height.")

    # Concatenate images horizontally
    new_image = Image.new('RGB', (img1.width + img2.width, img1.height))
    new_image.paste(img1, (0, 0))
    new_image.paste(img2, (img1.width, 0))

    # Save the concatenated image
    new_image.save(output_path)

def process_image(file, root, source_dir1, source_dir2, output_dir):
    if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        print(f"Skipping non-image file: {file}")
        return
        
    relative_path = os.path.relpath(root, source_dir2)
    output_subdir = os.path.join(output_dir, relative_path)
    
    # Create corresponding subdirectory in the output directory
    os.makedirs(output_subdir, exist_ok=True)
    
    base = os.path.join(source_dir1, relative_path, file)
    if os.path.exists(base):
        img1_path = base
    else:
        img1_path = os.path.join(source_dir1, relative_path, os.path.splitext(file)[0], "original_image.png")

    img2_path = os.path.join(root, file)
    output_img_path = os.path.join(output_subdir, file)

    if os.path.exists(img2_path):
        concatenate_images_horizontally(img1_path, img2_path, output_img_path)
    else:
        print(f"Warning: Corresponding image {img2_path} not found.")

def process_directory(source_dir1, source_dir2, output_dir):
    # List to collect all image files to process
    image_tasks = []
    
    for root, dirs, files in os.walk(source_dir2):
        if "process" in root:
            continue

        relative_path = os.path.relpath(root, source_dir2)
        output_subdir = os.path.join(output_dir, relative_path)
        
        # Create corresponding subdirectory in the output directory
        os.makedirs(output_subdir, exist_ok=True)
        
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_tasks.append((file, root))
    
    # Create a partial function with fixed parameters
    process_func = partial(
        process_image,
        source_dir1=source_dir1,
        source_dir2=source_dir2,
        output_dir=output_dir
    )
    
    # Determine the number of processes to use
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    
    # Process images in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        list(tqdm(
            pool.starmap(
                process_func, 
                [(file, root) for file, root in image_tasks]
            ),
            total=len(image_tasks),
            desc="Processing images"
        ))
    
    print(f"Processed {len(image_tasks)} images using {num_processes} processes")

def main():
    parser = argparse.ArgumentParser(description="Concatenate corresponding images from two directories horizontally.")
    parser.add_argument("--source-dir1", help="Original images", default="drag_data")
    parser.add_argument("--source-dir2", help="Dragged images", default="saved")
    parser.add_argument("--output-dir", help="Output directory where concatenated images will be saved", default="demo")

    args = parser.parse_args()

    if not os.path.isdir(args.source_dir1):
        print(f"Error: {args.source_dir1} is not a valid directory.")
        return

    if not os.path.isdir(args.source_dir2):
        print(f"Error: {args.source_dir2} is not a valid directory.")
        return

    process_directory(args.source_dir1, args.source_dir2, args.output_dir)
    print(f"Processing complete. Results saved in {args.output_dir}")

if __name__ == "__main__":
    main()