# GEMINI.md

## Project Overview

This project, "chess-board-analyzer", is a university computer vision assignment for La Rochelle Université. The ultimate goal is to create a Python application using OpenCV that can automatically analyze a video of a chess game and transcribe all the moves into standard algebraic notation.

The project is divided into several major parts as outlined in the project specification (`projet_echecs.pdf`):

1.  **Part 1: Board Detection and Rectification:** Detect the chessboard from a distorted perspective and transform it into a standardized, top-down square view.
2.  **Part 2: Player Hand/Move Detection:** Identify the frames in the video where a player is making a move.
3.  **Part 3: Piece Detection and Identification:** Determine the positions of all pieces on the board and identify them.
4.  **Part 4: Game Transcription:** Use the information from the previous steps to record the entire game in algebraic notation.

The `report/main.pdf` indicates that **Part 1 has been successfully implemented**. The current scripts focus on finding the board in an initial video (with an empty board), saving its corner coordinates, and creating a rectified view.

## Project Workflow & Current Status

The current workflow is focused on the initial setup and board detection:

1.  **Environment Setup (`uv sync`):** Installs dependencies like `opencv-python`.
2.  **Download Videos (`./download_videos.sh`):** Fetches the necessary input videos, including one of an empty board (`echiquier_vide_1.avi`) which is crucial for the first step.
3.  **Board Corner Detection (`python part1.py`):** This is the main script for the completed first part. It reads the empty board video, performs image preprocessing to find the board's contour, calculates the corner coordinates, and saves them to `corners_coordinates.json`. This file acts as a cache, "locking" the board's position for subsequent analysis of games played on it.
4.  **Output Generation (`part1.py`):** The script also generates processed videos (`output_videos/`) and summary images (`figures/`) that are used for visualization and in the project report.

## How to Run

### 1. Setup Environment

First, install the required Python packages using `uv`.

```bash
uv sync
```

### 2. Download Input Videos

The project requires video files to be present in the `input_videos/` directory. The provided script downloads them from Google Drive.

```bash
chmod +x ./download_videos.sh
./download_videos.sh
```

### 3. Run Board Detection & Rectification (Part 1)

Execute the main script for the first part of the analysis. This step only needs to be run once per physical setup, as it generates the `corners_coordinates.json` file.

```bash
python part1.py
```

This will produce:
-   `corners_coordinates.json`: The key output containing the board's corner locations.
-   `output_videos/part1_rectified.avi`: A video showing the corrected, top-down view of the board.
-   `figures/part1_summary.png`: A summary image visualizing the pipeline.

## Future Work (Next Steps)

Based on the project specification, the next steps involve:

-   **Implementing Hand Detection:** Create a new script or extend the existing one to analyze game videos (e.g., `mat_du_lion.avi`) and detect when hands enter the scene to make a move.
-   **Implementing Piece Recognition:** Develop logic to analyze the board state before and after a hand movement to determine which piece moved.
-   **Building the Transcription Logic:** Convert the detected move (e.g., "piece X from square A to square B") into algebraic notation.
-   **Integrating Advanced Methods:** The project allows for using Deep Learning (e.g., OpenCV's DNN module) to potentially improve piece or hand detection.

## Key Files

-   `projet_echecs.pdf`: The official project assignment document (in French).
-   `report/main.pdf`: The student's current progress report, detailing the successful implementation of Part 1.
-   `part1.py`: The main script for **Part 1**, responsible for board detection, rectification, and saving corner coordinates.
-   `preprocessing.py`: A utility script to generate a visualization of the image preprocessing steps for the report.
-   `corners_coordinates.json`: **Crucial output file.** Caches the board's corner coordinates, so detection doesn't need to run every time.
-   `download_videos.sh`: Script to download the required input videos.
-   `pyproject.toml`: Defines project metadata and Python dependencies.
