import { useState } from 'react';
import { Link } from 'react-router-dom';
import Viewer from '../components/Viewer.jsx';
import { API_BASE } from '../api.js';

export default function Extractor() {
  const [file, setFile] = useState(null);
  const [imageSrc, setImageSrc] = useState(null);
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selected = e.target.files?.[0];
    if (!selected) return;

    setFile(selected);
    setError(null);
    setResponse(null);

    if (selected.type === 'application/pdf') {
      setImageSrc(null);
      setError(
        'PDF files are not supported by the backend yet. Please upload an image (JPEG/PNG).'
      );
      return;
    }

    if (selected.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (event) => setImageSrc(event.target.result);
      reader.readAsDataURL(selected);
    } else {
      setImageSrc(null);
      setError('Please upload an image file (JPEG/PNG).');
    }
  };

  const handleAnalyzeDocument = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('image', file);

      const result = await fetch(`${API_BASE}/extract-document`, {
        method: 'POST',
        body: formData,
      });

      if (!result.ok) {
        let detail = `Server responded with status: ${result.status}`;
        try {
          const body = await result.json();
          if (body.detail) detail = body.detail;
        } catch {
          // ignore
        }
        throw new Error(detail);
      }

      const json = await result.json();
      setResponse(json);
      // Overlays are normalized against the perspective-corrected image, so
      // render that image (not the raw file preview) to keep boxes aligned.
      if (json.processed_image) setImageSrc(json.processed_image);
    } catch (err) {
      console.error('Failed to analyze document:', err);
      setError(
        err.message || 'Something went wrong while communicating with the server.'
      );
    } finally {
      setLoading(false);
    }
  };

  const copyJson = async () => {
    if (!response?.data) return;
    try {
      await navigator.clipboard.writeText(JSON.stringify(response.data, null, 2));
    } catch {
      // clipboard unavailable
    }
  };

  return (
    <div className="max-w-7xl mx-auto p-6">
      <header className="mb-6">
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">
              Document Data Extractor
            </h1>
            <p className="text-gray-600 mt-1">
              CV locates fields · LLM interprets meaning. Hover anything to
              see its bounding box on the document.
            </p>
          </div>
          <button
            onClick={copyJson}
            disabled={!response?.data}
            className="inline-flex items-center justify-center px-4 py-2 rounded-md text-sm font-semibold text-gray-700 bg-white border border-gray-300 hover:bg-gray-100 disabled:opacity-40 disabled:cursor-not-allowed shadow-sm"
          >
            Copy JSON
          </button>
        </div>

        <div className="mt-4 bg-white p-6 rounded-lg shadow-sm border border-gray-200 flex flex-col md:flex-row gap-4 items-start md:items-center justify-between">
          <div className="flex-1 w-full">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Document Image
            </label>
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
            />
            <p className="text-xs text-gray-400 mt-1.5">
              JPEG or PNG. PDF is not supported by the backend.
            </p>
          </div>

          <button
            onClick={handleAnalyzeDocument}
            disabled={!file || loading}
            className={`flex items-center justify-center px-6 py-2.5 rounded-md font-semibold text-white transition-all ${
              !file || loading
                ? 'bg-blue-300 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700 shadow-md'
            }`}
          >
            {loading ? (
              <>
                <svg
                  className="animate-spin -ml-1 mr-2 h-5 w-5 text-white"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  />
                </svg>
                Processing...
              </>
            ) : (
              'Analyze Document'
            )}
          </button>
        </div>

        {error && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
            <span className="font-bold">Error: </span> {error}
          </div>
        )}

        {response?.document_id && (
          <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg text-sm text-green-800 flex flex-wrap items-center justify-between gap-3">
            <span>
              Extraction saved to the dashboard.{' '}
              <Link
                to={`/document/${response.document_id}`}
                className="font-semibold underline hover:text-green-900"
              >
                Open full-page view →
              </Link>
            </span>
            <Link
              to="/"
              className="text-xs font-semibold text-green-700 bg-white border border-green-200 rounded-md px-3 py-1.5 hover:bg-green-100"
            >
              Go to dashboard
            </Link>
          </div>
        )}
      </header>

      {response && <Viewer response={response} imageSrc={imageSrc} />}
    </div>
  );
}