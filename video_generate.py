#!/usr/bin/env python3
"""
Video Generation Script using Vertex AI

This script generates videos using Vertex AI API with prompts from CSV files
and reference images from Brooklyn or Hoover directories.

Usage:
    python video_generate.py --csv BROOKLYN_BRIDGE_API_Ready.csv --count 5
    python video_generate.py --csv HOOVER_DAM_API_Ready.csv --count 10
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import vertexai
from vertexai.preview.vision_models import VideoGenerationModel
from langfuse import Langfuse
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize Langfuse client
langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_BASE_URL")
)

VERTEX_API_KEY = os.getenv("VERTEX_API_KEY")


def get_image_directory(csv_filename):
    """Determine the image directory based on CSV filename."""
    csv_name = Path(csv_filename).stem.lower()
    if "brooklyn" in csv_name:
        return "Brooklyn"
    elif "hoover" in csv_name:
        return "Hoover"
    else:
        raise ValueError(f"Cannot determine image directory for CSV: {csv_filename}")


def find_reference_image(image_dir, scene, shot_type, shot_title):
    """
    Find the reference image based on scene, shot type, and shot title.
    Image naming pattern: {order} - {shot_type} - {shot_title}.png
    """
    # Clean up shot_type for matching
    shot_type_clean = shot_type.replace("-ROLL", "")

    # Try exact match first
    exact_pattern = f"{scene} - {shot_type_clean} - {shot_title}.png"
    exact_path = Path(image_dir) / exact_pattern

    if exact_path.exists():
        return str(exact_path)

    # Try case-insensitive search
    image_dir_path = Path(image_dir)
    for img_file in image_dir_path.glob("*.png"):
        img_name = img_file.stem.lower()
        pattern = f"{scene.lower()} - {shot_type_clean.lower()} - {shot_title.lower()}"
        if pattern in img_name:
            return str(img_file)

    raise FileNotFoundError(
        f"Reference image not found for Scene {scene}, {shot_type}, {shot_title}\n"
        f"Expected pattern: {exact_pattern}\n"
        f"Directory: {image_dir}"
    )


def generate_video(prompt, reference_image_path, output_path, scene, shot_type, shot_title, project_id="your-project-id", location="us-central1"):
    """
    Generate a video using Vertex AI with the given prompt and reference image.

    Args:
        prompt: The text prompt for video generation
        reference_image_path: Path to the reference image
        output_path: Path where the generated video will be saved
        scene: Scene number for tracking
        shot_type: Shot type for tracking
        shot_title: Shot title for tracking
        project_id: Google Cloud project ID
        location: Vertex AI location
    """
    # Create a new trace in Langfuse for this video generation
    trace = langfuse.trace(
        name="video_generation",
        metadata={
            "scene": scene,
            "shot_type": shot_type,
            "shot_title": shot_title,
            "reference_image": reference_image_path,
            "output_path": output_path,
            "project_id": project_id,
            "location": location,
            "model": "veo-3.1-generate-001",
            "timestamp": datetime.now().isoformat()
        }
    )

    # Create a generation span for the API call
    generation = trace.generation(
        name="vertex_ai_video_generation",
        model="veo-3.1-generate-001",
        model_parameters={
            "aspect_ratio": "16:9",
            "duration_seconds": 4,
            "number_of_videos": 1
        },
        input={
            "prompt": prompt,
            "reference_image": reference_image_path
        }
    )

    print(f"Generating video with prompt: {prompt[:100]}...")
    print(f"Reference image: {reference_image_path}")
    print(f"Output: {output_path}")

    try:
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)

        # Load the video generation model
        model = VideoGenerationModel.from_pretrained("veo-3.1-generate-001")

        # Generate the video
        outputs = model.generate_video(
            prompt=prompt,
            reference_image=reference_image_path,
            number_of_videos=1,
            aspect_ratio="16:9",
            duration_seconds=4,
        )

        # Save the generated video
        for video in outputs:
            video.save(output_path)
            print(f"Video saved successfully to: {output_path}")

        # Update the generation with successful completion
        generation.end(
            output={
                "status": "success",
                "video_path": output_path,
                "video_generated": True
            }
        )

        # Update trace with success status
        trace.update(
            output={
                "status": "success",
                "video_path": output_path
            },
            level="SUCCESS"
        )

        return output_path

    except Exception as e:
        # Update the generation with error information
        generation.end(
            output={
                "status": "error",
                "error_message": str(e),
                "video_generated": False
            }
        )

        # Update trace with error status
        trace.update(
            output={
                "status": "error",
                "error": str(e)
            },
            level="ERROR"
        )

        raise e


def main():
    parser = argparse.ArgumentParser(
        description="Generate videos using Vertex AI API with prompts from CSV"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to the CSV file containing prompts (e.g., BROOKLYN_BRIDGE_API_Ready.csv)"
    )
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="Number of videos to generate"
    )
    parser.add_argument(
        "--project-id",
        type=str,
        default="your-project-id",
        help="Google Cloud project ID (default: your-project-id)"
    )
    parser.add_argument(
        "--location",
        type=str,
        default="us-central1",
        help="Vertex AI location (default: us-central1)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated_videos",
        help="Output directory for generated videos (default: generated_videos)"
    )

    args = parser.parse_args()

    # Validate CSV file exists
    if not Path(args.csv).exists():
        print(f"Error: CSV file not found: {args.csv}")
        return 1

    # Get the image directory
    try:
        image_dir = get_image_directory(args.csv)
        print(f"Using image directory: {image_dir}")
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Validate image directory exists
    if not Path(image_dir).exists():
        print(f"Error: Image directory not found: {image_dir}")
        return 1

    # Read CSV file
    print(f"Reading prompts from: {args.csv}")
    df = pd.read_csv(args.csv)

    # Validate count
    if args.count > len(df):
        print(f"Warning: Requested {args.count} videos but CSV only has {len(df)} rows")
        args.count = len(df)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Generate videos
    print(f"\nStarting video generation for {args.count} videos...\n")

    for i in range(min(args.count, len(df))):
        row = df.iloc[i]

        scene = str(row['Scene']).strip()
        shot_type = str(row['Shot_Type']).strip()
        shot_title = str(row['Shot_Title']).strip()
        prompt = str(row['Full_Prompt (Copy & Paste Ready)']).strip()

        print(f"\n{'='*60}")
        print(f"Video {i+1}/{args.count}")
        print(f"Scene: {scene}")
        print(f"Shot Type: {shot_type}")
        print(f"Shot Title: {shot_title}")
        print(f"{'='*60}")

        try:
            # Find reference image
            ref_image = find_reference_image(image_dir, scene, shot_type, shot_title)
            print(f"Found reference image: {ref_image}")

            # Generate output filename
            safe_shot_title = shot_title.replace(" ", "-").replace("/", "-")
            output_filename = f"{scene}_{shot_type}_{safe_shot_title}.mp4"
            output_path = output_dir / output_filename

            # Check if video already exists
            if output_path.exists():
                print(f"Video already exists, skipping: {output_path}")
                continue

            # Generate video
            generate_video(
                prompt=prompt,
                reference_image_path=ref_image,
                output_path=str(output_path),
                scene=scene,
                shot_type=shot_type,
                shot_title=shot_title,
                project_id=args.project_id,
                location=args.location
            )

            print(f"✓ Successfully generated: {output_filename}")

        except FileNotFoundError as e:
            print(f"✗ Error: {e}")
            continue
        except Exception as e:
            print(f"✗ Error generating video: {e}")
            continue

    print(f"\n{'='*60}")
    print("Video generation complete!")
    print(f"Videos saved to: {output_dir}")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    exit(main())
