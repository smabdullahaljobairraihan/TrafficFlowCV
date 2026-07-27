# TrafficFlowCV

TrafficFlowCV is a CPU-based traffic video analytics application developed using classical computer vision.

The application detects moving regions, tracks their centroids, estimates directional counting-line crossings, and generates downloadable traffic records, summaries, charts, and annotated videos.

## Live Demo

Try the deployed application:

[Open TrafficFlowCV](https://trafficflowcv-raihan.streamlit.app/)

## Features

- Traffic-video upload and preview
- Video metadata extraction
- MOG2 background subtraction
- Morphological motion-mask processing
- Contour-based moving-region detection
- Custom centroid-based object tracking
- Road Region of Interest filtering
- Upward and downward line-crossing estimation
- Pandas crossing-event and interval tables
- Matplotlib traffic charts
- Processed-video download
- Crossing-record CSV download
- Traffic-summary CSV download
- Interval-count CSV download
- Streamlit web interface
- Public deployment using Streamlit Community Cloud

## Application Interface

The Streamlit application is divided into three main sections:

### Upload & Analyze

- Upload a traffic video
- View video metadata
- Configure the analysis settings
- Run the complete traffic-video analysis

### Results Dashboard

- View estimated traffic crossings
- View upward and downward counts
- Play and download the processed video
- Inspect crossing-event records
- View interval-based traffic analysis
- View Matplotlib charts
- Download CSV reports

### Technical Preview

- Inspect a selected video frame
- Generate the foreground motion mask
- View candidate moving-region bounding boxes
- Generate a centroid-tracking preview

## Technology Stack

- Python
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Streamlit
- Git and GitHub
- Streamlit Community Cloud

## Processing Pipeline

1. Read the uploaded video frame by frame.
2. Learn the stationary background using OpenCV MOG2.
3. Extract moving foreground regions.
4. Clean the motion mask using filtering and morphological operations.
5. Restrict processing to the roadway Region of Interest.
6. Find contours inside the cleaned motion mask.
7. Filter small or unsuitable moving regions.
8. Create bounding boxes around accepted regions.
9. Calculate the centroid of each bounding box.
10. Match centroids between consecutive frames.
11. Assign and maintain tracking IDs.
12. Detect upward or downward counting-line crossings.
13. Store crossing events using Pandas.
14. Group crossings into time intervals.
15. Generate traffic charts using Matplotlib.
16. Display and export results through Streamlit.

## Detection and Tracking Method

TrafficFlowCV does not use YOLO or another trained deep-learning detector.

The system uses a classical computer-vision approach:

- OpenCV MOG2 for background subtraction
- Thresholding and morphological operations for motion-mask cleaning
- Contour analysis for moving-region detection
- Bounding-box centroids for object localization
- A custom centroid tracker for assigning IDs
- Euclidean distance for matching objects across consecutive frames

The detected objects represent moving regions inside the road ROI. They are not semantically classified as cars, buses, trucks, or motorcycles.

## Analysis Settings

The application provides the following settings in the sidebar:

### Minimum Moving-Region Area

Removes small motion regions that are likely to be noise.

### Road ROI Starting Position

Defines where the roadway Region of Interest begins vertically in the frame.

### Counting-Line Position

Defines the horizontal line used to detect upward and downward crossings.

### Background-Learning Frames

Defines how many initial frames are used to learn the stationary background.

### Analytics Time Interval

Defines the interval duration used for traffic-summary tables and charts.

## Project Structure

```text
TrafficFlowCV/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── sample_videos/
└── src/
    ├── __init__.py
    ├── analytics.py
    ├── centroid_tracker.py
    ├── motion_detector.py
    ├── object_detector.py
    ├── tracking_processor.py
    ├── vehicle_counter.py
    ├── video_processor.py
    └── video_tracker.py
```

## Module Responsibilities

### `app.py`

Provides the Streamlit user interface, sidebar settings, application tabs, processing controls, result display, and downloads.

### `video_processor.py`

Saves uploaded videos, extracts metadata, and reads individual frames.

### `motion_detector.py`

Performs background subtraction and motion-mask cleaning.

### `object_detector.py`

Finds contours, filters moving regions, and creates bounding boxes and centroids.

### `centroid_tracker.py`

Matches centroids between consecutive frames and assigns tracking IDs.

### `tracking_processor.py`

Generates a technical tracking preview for a selected portion of the video.

### `vehicle_counter.py`

Detects upward and downward counting-line crossings and prevents duplicate counting for the same tracked ID.

### `video_tracker.py`

Combines ROI filtering, motion detection, object detection, centroid tracking, crossing detection, annotation, and output-video generation.

### `analytics.py`

Uses Pandas and NumPy to prepare traffic tables and summary statistics, and Matplotlib to generate traffic charts.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/smabdullahaljobairraihan/TrafficFlowCV.git
cd TrafficFlowCV
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install the dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
python -m streamlit run app.py
```

Open the local address displayed in the terminal, usually:

```text
http://localhost:8501
```

## Recommended Video Conditions

For better results, use videos with:

- A fixed camera
- A clearly visible roadway
- Moderate traffic density
- Stable lighting
- Limited vehicle overlap
- Limited camera vibration
- Vehicles moving through the counting-line area
- Short duration
- MP4 format
- Resolution of 720p or lower

## Recommended and motion-mask cleaning.

### `object_detector.py`

Finds contours, filters moving regions, and creates bounding boxes and centroids.

### `centroid_tracker.py`

Matches centroids between consecutive frames and assigns tracking IDs.

### `tracking_processor.py`

Generates a technical tracking preview for a selected portion of the video.

### `vehicle_counter.py`

Detects upward and downward counting-line crossings and prevents duplicate counting for the same tracked ID.

### `video_tracker.py`

Combines ROI filtering, motion detection, object detection, centroid tracking, crossing detection, annotation, and output-video generation.

### `analytics.py`

Uses Pandas and NumPy to prepare traffic tables and summary statistics, and Matplotlib to generate traffic charts.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/smabdullahaljobairraihan/TrafficFlowCV.git
cd TrafficFlowCV
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install the dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
python -m streamlit run app.py
```

Open the local address displayed in the terminal, usually:

```text
http://localhost:8501
```

## Recommended Video Conditions

For better results, use videos with:

- A fixed camera
- A clearly visible roadway
- Moderate traffic density
- Stable lighting
- Limited vehicle overlap
- Limited camera vibration
- Vehicles moving through the counting-line area
- Short duration
- MP4 format
- Resolution of 720p or lower

## Recommended Initial Settings

```text
Minimum moving-region area: 500
Road ROI Initial Settings

```text
Minimum moving-region area: 500
Road ROI starting position: 35%
Counting-line position: 70%
Background-learning frames starting position: 35%
Counting-line position: 70%
Background-learning frames: 90
Analytics interval: 5 seconds
```

The: 90
Analytics interval: 5 seconds
```

The ideal settings may vary depending ideal settings may vary depending on the camera position, road geometry, traffic density, on the camera position, road geometry, traffic density, and video resolution.

## Outputs

The application can and video resolution.

## Outputs

The application can generate:

- Annotated processed video
- Estimated generate:

- Annotated processed video
- Estimated total crossing count
- Up total crossing count
- Upward crossing count
- Downward crossing count
- Crossing-event table
- Time-interval traffic table
- Cward crossing count
- Downward crossing count
- Crossing-event table
- Time-interval traffic table
- Cumulative traffic chart
- Direction-comparison chart
-umulative traffic chart
- Direction-comparison chart
- Traffic-summary CSV
- Crossing-record CSV
- Interval-count CSV

## Traffic-summary CSV
- Crossing-record CSV
- Interval-count CSV

## Limitations

TrafficFlowCV uses classical motion detection instead Limitations

TrafficFlowCV uses classical motion detection instead of a trained vehicle detector.

As a result:

- Moving of a trained vehicle detector.

As a result:

- Moving shadows may be detected.
- Nearby vehicles may merge into one region.
- One vehicle shadows may be detected.
- Nearby vehicles may merge into one region.
- One vehicle may be divided into multiple regions.
- Occlusion may be divided into multiple regions.
- Occlusion may cause tracking IDs may cause tracking IDs to be lost.
- A lost vehicle may receive a new ID.
- to be lost.
- A lost vehicle may receive a new ID.
- Camera movement may produce Camera movement may produce false detections.
- Dense traffic can false detections.
- Dense traffic can reduce tracking accuracy.
- The system does not identify reduce tracking accuracy.
- The system does not identify vehicle classes.

Therefore, the generated counts should be treated as experimental traffic-flow vehicle classes.

Therefore, the generated counts should be treated as experimental traffic-flow estimates rather than verified ground-truth vehicle counts.

## Future Improvements

- YOLO-based vehicle detection
- ByteTrack or BoT-SORT integration
- Vehicle-class identification
- Lane-specific estimates rather than verified ground-truth vehicle counts.

## Future Improvements

- YOLO-based vehicle detection
- ByteTrack or vehicle counting
- Perspective-aware detection thresholds
- Improved shadow removal
- Better BoT-SORT integration
- Vehicle-class identification
- Lane-specific vehicle counting
- Perspective-aware detection thresholds
- Improved shadow removal
- Better occlusion handling
- User-defined polygonal occlusion handling
- User-defined polygonal ROI
- User-defined counting-line ROI
- User-defined counting-line endpoints
- Database endpoints
- Database-based result storage
- Large-video processing support

## Project Purpose

This project was-based result storage
- Large-video processing support

## Project Purpose

This project was developed as a practical demonstration of:

- Python modular programming
- Classical developed as a practical demonstration of:

- Python modular programming
- Classical computer vision
- OpenCV video processing
- NumPy numerical operations
- Pandas data analysis
- Matplotlib visualization
- Streamlit computer vision
- OpenCV video processing
- NumPy numerical operations
- Pandas data analysis
- Matplotlib visualization
- Streamlit application development
- GitHub version control
- Cloud application development
- GitHub version control
- Cloud deployment