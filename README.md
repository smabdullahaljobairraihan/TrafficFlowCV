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
- Public deployment through Streamlit Community Cloud

## Application Interface

The Streamlit application is divided into three main sections.

### Upload & Analyze

This section allows the user to:

- Upload a traffic video
- View video metadata
- Review the selected analysis settings
- Run the complete traffic-video analysis

### Results Dashboard

This section displays:

- Estimated total crossings
- Upward crossing count
- Downward crossing count
- Processed traffic video
- Crossing-event records
- Time-interval traffic analysis
- Matplotlib charts
- Video and CSV downloads

### Technical Preview

This optional section allows the user to:

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
- Git
- GitHub
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
16. Display and export the results through Streamlit.

## Detection and Tracking Method

TrafficFlowCV does not use YOLO or another trained deep-learning detector.

The application uses a classical computer-vision approach:

- OpenCV MOG2 for background subtraction
- Thresholding for foreground extraction
- Median filtering for noise reduction
- Morphological opening and closing for mask cleaning
- Dilation for connecting fragmented motion regions
- Contour analysis for moving-region detection
- Bounding-box centroids for object localization
- A custom centroid tracker for assigning IDs
- Euclidean distance for matching objects between consecutive frames

The detected objects represent moving regions inside the road ROI. They are not semantically classified as cars, buses, trucks, or motorcycles.

## Region of Interest

ROI means Region of Interest.

The application creates a polygonal roadway region and ignores motion outside that area. This helps reduce false detections from the sky, buildings, trees, sidewalks, and other irrelevant parts of the frame.

## Counting-Line Method

A horizontal counting line is placed inside the selected roadway ROI.

For each tracked object, the application compares the previous and current centroid positions relative to the line.

A crossing is recorded when:

- A centroid moves from above the line to below the line
- A centroid moves from below the line to above the line

Each tracked ID is counted only once.

## Analysis Settings

The sidebar contains the following controls.

### Minimum Moving-Region Area

This setting removes small motion regions that are likely to be noise.

A higher value removes more small regions, while a lower value allows smaller or more distant moving regions to be detected.

### Road ROI Starting Position

This setting defines where the roadway Region of Interest begins vertically in the video frame.

The percentage is measured from the top of the frame.

### Counting-Line Position

This setting defines the vertical position of the horizontal counting line.

The percentage is measured from the top of the processed frame.

### Background-Learning Frames

This setting determines how many initial frames are used to learn the stationary background.

These initial frames are not used for counting.

### Analytics Time Interval

This setting determines the duration used to group crossing events into traffic-analysis intervals.

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

Provides the Streamlit user interface, sidebar settings, application tabs, processing controls, result display, charts, and downloads.

### `video_processor.py`

Handles:

- Saving uploaded videos
- Extracting video metadata
- Reading individual video frames

### `motion_detector.py`

Handles:

- MOG2 background subtraction
- Foreground thresholding
- Median filtering
- Morphological opening
- Morphological closing
- Dilation
- Removal of small motion regions

### `object_detector.py`

Handles:

- Contour detection
- Moving-region filtering
- Bounding-box creation
- Centroid calculation
- Bounding-box annotation

### `centroid_tracker.py`

Handles:

- Centroid-distance calculation
- Matching objects between consecutive frames
- Tracking-ID assignment
- Temporary object disappearance
- Lost-object deregistration

### `tracking_processor.py`

Generates a technical tracking preview for a selected section of the video.

### `vehicle_counter.py`

Handles:

- Determining whether an object is above or below the counting line
- Detecting upward and downward crossings
- Preventing duplicate counting
- Creating crossing-event records

### `video_tracker.py`

Combines:

- Video reading
- ROI creation
- Background subtraction
- Motion-mask cleaning
- Moving-region detection
- Centroid tracking
- Counting-line crossing detection
- Frame annotation
- Output-video generation

### `analytics.py`

Uses Pandas, NumPy, and Matplotlib to:

- Build the crossing-event table
- Group events into time intervals
- Calculate traffic-summary statistics
- Identify the peak traffic interval
- Generate traffic charts
- Prepare downloadable CSV reports

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

On Windows Command Prompt:

```bat
venv\Scripts\activate
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
- Vehicles crossing the selected counting line
- Short duration
- MP4 format
- Resolution of 720p or lower

## Recommended Initial Settings

```text
Minimum moving-region area: 500
Road ROI starting position: 35%
Counting-line position: 70%
Background-learning frames: 90
Analytics interval: 5 seconds
```

The ideal settings may vary depending on the camera position, road geometry, traffic density, lighting conditions, and video resolution.

## Outputs

The application can generate:

- Annotated processed video
- Estimated total crossing count
- Upward crossing count
- Downward crossing count
- Crossing-event table
- Time-interval traffic table
- Crossings-by-interval bar chart
- Cumulative traffic-count line chart
- Direction-comparison bar chart
- Traffic-summary CSV
- Crossing-record CSV
- Interval-count CSV

## Limitations

TrafficFlowCV uses classical motion detection instead of a trained vehicle detector.

As a result:

- Moving shadows may be detected.
- Nearby vehicles may merge into one moving region.
- One vehicle may be divided into multiple regions.
- Occlusion may cause tracking IDs to be lost.
- A lost vehicle may receive a new tracking ID.
- Camera movement may produce false detections.
- Dense traffic can reduce tracking accuracy.
- The system does not classify vehicle types.
- The same vehicle may occasionally receive more than one tracking ID.
- Large or high-resolution videos may require significant processing time.

Therefore, the generated counts should be treated as experimental traffic-flow estimates rather than verified ground-truth vehicle counts.

## Future Improvements

- YOLO-based vehicle detection
- ByteTrack or BoT-SORT integration
- Vehicle-class identification
- Lane-specific vehicle counting
- User-defined polygonal ROI
- User-defined counting-line endpoints
- Improved shadow removal
- Improved occlusion handling
- Perspective-aware detection thresholds
- Automatic parameter selection
- Database-based result storage
- Large-video processing support

## Project Purpose

This project demonstrates practical skills in:

- Python modular programming
- Classical computer vision
- OpenCV video processing
- NumPy numerical operations
- Pandas data analysis
- Matplotlib visualization
- Streamlit application development
- Git version control
- GitHub project management
- Cloud deployment

## Disclaimer

TrafficFlowCV is an educational and portfolio project.

It is not intended to replace professional traffic-monitoring systems or provide legally verified traffic-count data.