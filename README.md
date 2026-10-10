# Form Extraction Engine

A high-performance, deterministic document extraction engine that uses Computer Vision (OpenCV) and OCR (PaddleOCR) to extract key-value pairs from forms with extreme precision. 

Instead of relying on slow and unpredictable Large Language Models (LLMs), this engine uses a **Template-based Homography Alignment** pipeline. You register a blank "Template" once, and then any filled version of that form is warped and aligned to the template, allowing for lightning-fast, zero-shot structured extraction.

## Features
- **Template Registration**: Upload an empty form to automatically detect form fields, checkboxes, and tables using CV contours and line detection.
- **Homography Warping**: Filled forms are automatically deskewed, rotated, and geometrically warped to perfectly align with their parent template.
- **Lightning Fast Extraction**: Because the geometry is known ahead of time via the template, extraction happens purely via OCR intersection—no LLM parsing required.
- **Checkbox & Table Support**: Natively detects checked/unchecked states and structured grid data.
- **Clerk Authentication**: Secure frontend and backend authentication using Clerk JWTs.
- **Storage Agnostic**: Supports local SQLite/file storage for development, and AWS S3/PostgreSQL for production via a simple `STORAGE_MODE` toggle.
- **Batch Processing**: Background workers capable of processing large volumes of forms asynchronously.

## Architecture

The project is split into a separated backend and frontend:

* **Backend (`/`)**: A FastAPI application that orchestrates the computer vision pipeline.
  * `main.py`: The entrypoint for the FastAPI server.
  * `routers/`: Contains the API endpoints for `templates`, `filled_forms`, and `batch` processing.
  * `service/`: The core business logic. Includes `ocr_service.py` (PaddleOCR wrappers), `preprocess_service.py` (deskew/orientation), and `template_service.py` (alignment logic).
  * `util/`: Helper functions for geometry math and CV operations.
* **Frontend (`/ui`)**: A React application built with Vite and TailwindCSS.
  * Provides a modern dashboard to upload templates, view extracted data, and manage form batches.

---

## Getting Started (Local Development)

### 1. Prerequisites
- Python 3.10+
- Node.js 18+

### 2. Backend Setup
1. Navigate to the project root directory.
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure your environment variables. Create a `.env` file in the root directory:
   ```env
   STORAGE_MODE=local
   LOCAL_STORAGE_DIR=uploads
   LOCAL_DATABASE_URL=sqlite:///./data/form_extract.db
   CLERK_SECRET_KEY=your_clerk_secret_key
   CLERK_FRONTEND_API_URL=your_clerk_frontend_api
   ```
5. Start the backend server:
   ```bash
   uvicorn main:app --reload
   ```
   *The API will be available at `http://127.0.0.1:8000`.*

### 3. Frontend Setup
1. Open a new terminal and navigate to the `ui` directory:
   ```bash
   cd ui
   ```
2. Install Node dependencies:
   ```bash
   npm install
   ```
3. Configure frontend environment variables. Create a `.env` file in the `ui` directory:
   ```env
   VITE_CLERK_PUBLISHABLE_KEY=your_clerk_publishable_key
   ```
4. Start the Vite development server:
   ```bash
   npm run dev
   ```
   *The UI will be available at `http://localhost:5173`.*

---

## How to Use

1. **Log In**: Open the frontend (`http://localhost:5173`) and sign in using your Clerk account.
2. **Register a Template**: 
   - Navigate to the **Templates** tab.
   - Upload a **blank** (unfilled) version of the form you want to extract data from.
   - The engine will analyze the geometry and save it as a master template.
3. **Extract Data**:
   - Click on your newly created template to view its details.
   - Upload **filled** versions of that specific form.
   - The engine will warp the filled image to match the template, extract the text from the exact field locations, and instantly return the structured JSON data.
