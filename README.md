# TrafficFlowCV

TrafficFlowCV is a CPU-based traffic video analytics application developed using classical computer vision.

The application detects moving regions, tracks their centroids, estimates directional counting-line crossings, and produces downloadable traffic analytics.

## Features

- Traffic-video upload and preview
- Video metadata extraction
- MOG2 background subtraction
- Morphological motion-mask processing
- Contour-based moving-region detection
- Custom centroid-based object tracking
- Road Region of Interest filtering
- Upward and downward line-crossing estimation
- Pandas event and interval tables
- Matplotlib traffic charts
- Processed-video download
- CSV report downloads
- Streamlit web interface

## Technology Stack

- Python
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Streamlit

## Processing Pipeline

1. Read the uploaded video frame by frame.
2. Learn the stationary background using MOG2.
3. Extract moving foreground regions.
4. Clean the foreground mask using morphological operations.
5. Find contours and create bounding boxes.
6. Calculate bounding-box centroids.
7. Match centroids between consecutive frames.
8. Assign tracking IDs.
9. Restrict processing to the roadway ROI.
10. Record objects crossing the counting line.
11. Analyze crossing events using Pandas.
12. Generate traffic charts using Matplotlib.

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