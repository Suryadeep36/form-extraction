# Project Documentation: Deterministic Form Extraction Engine

## 1. Problem Statement
Extracting structured data (key-value pairs, checkboxes, tables) from scanned or photographed forms is a notoriously difficult problem. 
Historically, solutions fall into two categories:
1. **Rule-Based/Zonal OCR**: Extremely brittle. If the document is scanned at a slight angle or shifted by a few pixels, the hardcoded coordinates fail to capture the correct data.
2. **LLM-Based (Large Language Models)**: While LLMs can interpret unstructured data, they are expensive, prone to hallucination (inventing data), slow, and struggle with strict geometric layouts like tables or checkboxes.

## 2. Solution: Template-Based Homography Alignment
This project solves the form extraction problem by eliminating the reliance on LLMs and brittle hardcoded zones. Instead, it utilizes **Computer Vision (CV)** and **Homography Warping**.

**How it was solved:**
The engine operates on a two-pass "Template" system. 
1. **Pass 1 (Template Registration)**: A user uploads a blank (unfilled) version of a form. The system uses CV algorithms (Canny edge detection, Hough transforms, contour analysis) to automatically detect all input fields, checkboxes, and tables, establishing a baseline geometry.
2. **Pass 2 (Extraction)**: When a filled version of the form is uploaded, the system uses feature matching (ORB/SIFT) to calculate a homography matrix. It warps, rotates, and aligns the filled document perfectly over the original template. 
Because the documents are now perfectly aligned, the system simply runs OCR (Optical Character Recognition) on the predefined geometric boundaries from the template.

This results in **deterministic, lightning-fast, and highly accurate** extraction without using an LLM.

---

## 3. Core Features
- **Zero-Shot Deterministic Extraction**: Extracts data accurately without requiring model fine-tuning for new document types.
- **Homography Warping**: Automatically corrects skew, rotation, and perspective distortion on mobile uploads.
- **Native Checkbox Detection**: Uses structural analysis to detect checked vs. unchecked states without relying on OCR character misinterpretations.
- **Complex Table Parsing**: Detects table grids (both bordered and borderless) using line detection algorithms and accurately maps OCR text to distinct cells.
- **Batch Processing Engine**: Background workers queue and process massive volumes of forms asynchronously without blocking the main API.
- **Dual-Storage Engine**: Operates locally (SQLite + Local File System) for development, or entirely in the cloud (PostgreSQL + AWS S3) for production via a simple `STORAGE_MODE` toggle.
- **Enterprise-Grade Authentication**: Integrated with Clerk for secure JWT-based stateless authentication across the frontend and backend.

---

## 4. Architecture
The project follows a decoupled client-server architecture:

### Frontend (Client)
- Built with **React** and **Vite** for rapid compilation and modern component rendering.
- Styled using **TailwindCSS** for responsive, utility-first design.
- Uses **React Router** for seamless single-page application (SPA) navigation.
- Interfaces with **Clerk React SDK** to manage user sessions and inject JWTs into API requests.

### Backend (Server)
- Built with **FastAPI**, a high-performance asynchronous Python web framework.
- **Routers**: API endpoints are strictly divided by domain (`template_router.py`, `filled_form_router.py`, `batch_router.py`).
- **Services**: The business logic layer. 
  - `preprocess_service.py`: Handles orientation and deskewing.
  - `ocr_service.py`: Wraps PaddleOCR for text bounding box extraction.
  - `template_service.py`: Handles the OpenCV homography alignment and field mapping.
  - `blob_store.py`: Abstraction layer for saving images to S3 or local disk.
- **Stateless Operation**: The backend relies entirely on the database and S3 for state, making it horizontally scalable.

---

## 5. System Workflow

1. **User Authentication**: User logs in via Clerk on the React frontend.
2. **Template Registration**:
   - User uploads `blank_form.jpg`.
   - FastAPI receives the image.
   - `preprocess_service` deskews the image.
   - OpenCV detects horizontal/vertical lines to build a structural graph of input regions.
   - The detected fields and template image are saved to the database and S3.
3. **Data Extraction**:
   - User uploads `filled_form.jpg` and specifies the Template ID.
   - FastAPI loads the master template image.
   - OpenCV calculates the homography matrix between the filled image and the template image.
   - The filled image is warped to exactly match the template's coordinate system.
   - PaddleOCR extracts text from the document.
   - The system intersects the OCR bounding boxes with the template's known field coordinates to map values to labels.
4. **Results Delivery**: The structured JSON data is returned to the frontend and persisted in the database.

---

## 6. Models and Core Algorithms
- **PaddleOCR**: Used for highly accurate, multilingual text detection and recognition. PaddleOCR was chosen over Tesseract due to its superior performance on handwritten text and varied fonts.
- **OpenCV Homography (RANSAC)**: Used to compute the perspective transformation between two images, rejecting outliers (like handwritten text or stamps) to align the underlying structural features of the forms.
- **Canny Edge Detection & Hough Line Transform**: Used to detect the physical boxes, underlines, and table grids on blank templates.

---

## 7. Important Libraries
- **FastAPI**: Backend web framework.
- **Uvicorn**: ASGI server for running FastAPI.
- **OpenCV-Python (`cv2`)**: Core computer vision library for image manipulation, warping, and structural detection.
- **PaddleOCR & PaddlePaddle**: Machine learning frameworks for Optical Character Recognition.
- **Numpy**: Fast matrix math for coordinate transformations and bounding box calculations.
- **Boto3**: AWS SDK for Python, used for S3 object storage integration.
- **React & Vite**: Frontend UI library and build tool.
- **TailwindCSS**: Frontend styling framework.
- **Clerk (`@clerk/clerk-react`)**: Authentication provider.

---

## 8. Future Enhancements & Features

### 1. Advanced Verification UI (Human-in-the-Loop)
- **Feature**: Build a side-by-side verification interface where users can click on an extracted key-value pair and see a highlighted crop of the original document.
- **Benefit**: Allows operators to quickly review and correct low-confidence OCR reads before exporting the data.

### 2. Multi-Page Document Support
- **Feature**: Extend the homography engine to support multi-page PDFs.
- **Benefit**: Users can upload 10-page packets. The engine will automatically classify each page against a library of templates and extract data across the entire packet.

### 3. Anchor-Based Alignment
- **Feature**: Instead of warping the entire page (which can fail on highly distorted mobile photos), identify fixed anchors (like barcodes, logos, or specific text strings) and calculate local offsets for specific fields.
- **Benefit**: Higher resilience to extremely crumpled or folded paper documents.

### 4. Export Integrations
- **Feature**: Add one-click exports to CSV, Excel, Google Sheets, or webhooks.
- **Benefit**: Seamlessly integrate the extracted data into downstream ERP or CRM systems.

### 5. API Key Management
- **Feature**: Allow users to generate API keys from the dashboard to programmatically trigger batch extractions from their own backend systems.
