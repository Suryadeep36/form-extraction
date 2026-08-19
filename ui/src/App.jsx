import { useState } from 'react';
import DocumentImage from './components/DocumentImage.jsx';
import ExtractionPanel from './components/ExtractionPanel.jsx';
import StatsBar from './components/StatsBar.jsx';
import DebugPanel from './components/DebugPanel.jsx';
import { COLORS } from './bbox.js';

const LAYER_OPTIONS = [
  { key: 'sections', label: 'Sections', color: COLORS.section },
  { key: 'labels', label: 'Labels', color: COLORS.label },
  { key: 'values', label: 'Values', color: COLORS.value },
  { key: 'checkboxes', label: 'Checkboxes', color: COLORS.checkbox },
  { key: 'unassigned', label: 'Unassigned', color: COLORS.unassigned },
  { key: 'elements', label: 'OCR Elements', color: COLORS.element },
];

function Legend({ layers, onToggleLayer }) {
  return (
    <div className="flex flex-wrap items-center gap-2 mb-3">
      {LAYER_OPTIONS.map((option) => (
        <button
          key={option.key}
          onClick={() => onToggleLayer(option.key)}
          className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium border transition-colors ${
            layers[option.key]
              ? 'bg-white text-gray-800 border-gray-300 shadow-sm'
              : 'bg-gray-100 text-gray-400 border-gray-200'
          }`}
          title={`Toggle ${option.label} overlay`}
        >
          <span
            className="inline-block w-3 h-3 rounded-sm"
            style={{
              backgroundColor: layers[option.key] ? option.color : 'transparent',
              border: `2px solid ${option.color}`,
            }}
          />
          {option.label}
        </button>
      ))}
    </div>
  );
}

export default function DocumentViewer() {
  const [file, setFile] = useState(null);
  const [imageSrc, setImageSrc] = useState(null);
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hovered, setHovered] = useState(null);
  const [layers, setLayers] = useState({
    sections: true,
    labels: true,
    values: true,
    checkboxes: true,
    unassigned: false,
    elements: false,
  });
  const [showDebug, setShowDebug] = useState(false);

  const data = response?.data;
  const rep = response?.document_representation;

  const handleFileChange = (e) => {
    const selected = e.target.files?.[0];
    if (!selected) return;

    setFile(selected);
    setError(null);
    setResponse(null);
    setHovered(null);

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
    setHovered(null);

    try {
      const formData = new FormData();
      formData.append('image', file);

      const result = await fetch('http://localhost:8000/extract-document', {
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
    } catch (err) {
      console.error('Failed to analyze document:', err);
      setError(
        err.message || 'Something went wrong while communicating with the server.'
      );
    } finally {
      setLoading(false);
    }
  };

  const toggleLayer = (key) =>
    setLayers((prev) => ({ ...prev, [key]: !prev[key] }));

  const copyJson = async () => {
    if (!data) return;
    try {
      await navigator.clipboard.writeText(JSON.stringify(data, null, 2));
    } catch {
      // clipboard unavailable
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6 font-sans text-gray-800">
      <div className="max-w-7xl mx-auto">
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
              disabled={!data}
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
        </header>

        {response && (
          <div className="flex flex-col gap-6 animate-fade-in-up">
            <StatsBar response={response} />

            <Legend layers={layers} onToggleLayer={toggleLayer} />

            <div className="flex flex-col lg:flex-row gap-6">
              <div className="w-full lg:w-[52%]">
                <div className="flex justify-between items-center mb-3">
                  <h2 className="text-lg font-bold">Document Image</h2>
                  <span className="text-xs text-gray-400">
                    blue=section · red=label · green=value · purple=checkbox ·
                    orange=unassigned
                  </span>
                </div>
                <DocumentImage
                  imageSrc={imageSrc}
                  sections={data?.sections || []}
                  unassigned={data?.unassigned_text || []}
                  elements={rep?.elements || []}
                  imageSize={response.image}
                  layers={layers}
                  hovered={hovered}
                />
              </div>

              <div className="w-full lg:flex-1">
                <div className="flex items-center justify-between mb-3">
                  <h2 className="text-lg font-bold">Extracted Information</h2>
                  {data?.document_type && (
                    <span className="text-sm text-gray-500">
                      Type:{' '}
                      <span className="font-semibold text-gray-700">
                        {data.document_type}
                      </span>
                    </span>
                  )}
                </div>
                <div className="h-full lg:max-h-[820px] overflow-y-auto pr-1 pb-8 custom-scrollbar">
                  <ExtractionPanel data={data} onHover={setHovered} />
                </div>
              </div>
            </div>

            {rep && (
              <div className="flex justify-end">
                <button
                  onClick={() => setShowDebug((prev) => !prev)}
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-md text-sm font-semibold text-gray-700 bg-white border border-gray-300 hover:bg-gray-100 shadow-sm"
                >
                  {showDebug ? 'Hide' : 'Show'} Debug Panel
                </button>
              </div>
            )}

            {showDebug && (
              <DebugPanel
                response={response}
                imageSize={response.image}
                onHover={setHovered}
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}
