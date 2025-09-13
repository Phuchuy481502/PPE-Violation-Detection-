# CS406 PPE Violation Detection
- Professor: ThS. Cáp Phạm Đình Thăng (thangcpd@uit.edu.vn)
- Course Id: CS406.P11 (Xử lý ảnh và ứng dụng)

## Overview
Simple full pipeline object detection (PPE detection) and tracking.  
This project focuses on detecting PPE (Hardhat, Helmet, Gloves) and classifying each person if they are missing one of these.

## Team Members
| Name                | MSSV      | Roles  |
|---------------------|-----------|--------|
| Nguyễn Hữu Nam      | 22520917  | Leader |
| Nguyễn Trần Phúc    | 22521135  | Member |
| Hồ Trọng Duy Quang  | 22521200  | Member |

## A glimpse of the project
<div align="center">
  <img src="assets/1.png" alt="Project Image" width="600"/>
  <p><em>Figure 1: Overview of how Violation Detection works on image</em></p>
</div>

<div align="center">
  <img src="assets/2.png" alt="Project Image" width="600"/>
  <p><em>Figure 2: Overview of how Violation Detection works on video</em></p>
</div>

<div align="center">
  <img src="assets/3.png" alt="Project Image" width="600"/>
  <p><em>Figure 3: Project pipeline</em></p>
</div>

<div align="center">
  <img src="assets/4.png" alt="Project Image" width="600"/>
  <p><em>Figure 4: Web demo</em></p>
</div>

> **Overview**: this project focus on detecting PPE (Hardhat, Helmet, Gloves) and classify each person if they are missing one of these  
- *Image*: detect + classify violation  
- *Video*: detect + track + classify violation (video result: [drive](https://drive.google.com/drive/folders/15crPWioDnb8FuSfXhSJFO__1lEhy6D6j?usp=sharing))

# Project structure
```python
CS406-PPE-detection/
├── data/
│   ├── data-ppe.yaml
│   └── split/      #contain train, val, test
├── logs/
├── notebooks/
├── output/
├── sample/
├── scripts/
│   ├── detect_faster_rcnn.py
│   ├── detect_yolo.py
│   ├── loader_faster_rcnn.py
│   └── tracker_yolo.py
├── src/
│   ├── loader/
│   ├── models/
│   ├── parsers/
│   ├── trackers/
│   └── utils/
├── tools/
├── web/
│   ├── app.py
│   └── output/     #output of web
├── weights/
│   ├── best_faster_rcnn.pt
│   └── best_yolo.pt
├── README.md
├── requirements.txt
└── setup.py
