import { useState } from 'react';
import DocumentImage from './DocumentImage.jsx';
import ExtractionPanel from './ExtractionPanel.jsx';
import StatsBar from './StatsBar.jsx';
import DebugPanel from './DebugPanel.jsx';
import { COLORS } from '../bbox.js';

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

export default function Viewer({ response, imageSrc }) {
  const [hovered, setHovered] = useState(null);
  const [showDebug, setShowDebug] = useState(false);
  const [layers, setLayers] = useState({
    sections: true,
    labels: true,
    values: true,
    checkboxes: true,
    unassigned: false,
    elements: false,
  });

  const data = response?.data;
  const rep = response?.document_representation;

  const toggleLayer = (key) =>
    setLayers((prev) => ({ ...prev, [key]: !prev[key] }));

  return (
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
            imageSize={response?.image}
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
          imageSize={response?.image}
          onHover={setHovered}
        />
      )}
    </div>
  );
}