import { COLORS } from '../bbox.js';

function ElementRow({ element, imageSize, onHover }) {
  return (
    <tr
      className="hover:bg-cyan-50 cursor-default transition-colors"
      onMouseEnter={() =>
        onHover({
          type: 'element',
          key: element.id,
          boxes: [
            {
              bbox: element.bbox
                ? [
                    element.bbox[0] / imageSize.width,
                    element.bbox[1] / imageSize.height,
                    element.bbox[2] / imageSize.width,
                    element.bbox[3] / imageSize.height,
                  ]
                : null,
              color: COLORS.element,
            },
          ],
          label: `${element.id}: ${element.text}`,
        })
      }
      onMouseLeave={() => onHover(null)}
    >
      <td className="px-3 py-1 font-mono text-xs text-gray-500 whitespace-nowrap">
        {element.id}
      </td>
      <td className="px-3 py-1 text-sm text-gray-800">{element.text}</td>
      <td className="px-3 py-1 font-mono text-[10px] text-gray-400 whitespace-nowrap">
        {element.bbox?.join(', ')}
      </td>
      <td className="px-3 py-1 text-xs text-gray-500">
        {element.confidence == null
          ? '—'
          : (element.confidence * 100).toFixed(0) + '%'}
      </td>
    </tr>
  );
}

export default function DebugPanel({ response, imageSize, onHover }) {
  const rep = response?.document_representation;
  const llmOutput = response?.llm_output;

  if (!rep && !llmOutput) return null;

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200">
        <h3 className="font-semibold text-gray-800">
          Debug / Traceability
        </h3>
        <span className="text-xs text-gray-400">
          Hover rows to highlight on the document
        </span>
      </div>

      <details className="px-4 py-3 border-b border-gray-100">
        <summary className="cursor-pointer text-sm font-medium text-gray-700">
          OCR Elements ({rep?.elements?.length ?? 0})
        </summary>
        <div className="mt-2 max-h-72 overflow-y-auto custom-scrollbar">
          <table className="w-full text-left">
            <thead className="bg-gray-50 text-[10px] uppercase tracking-wider text-gray-400">
              <tr>
                <th className="px-3 py-1">ID</th>
                <th className="px-3 py-1">Text</th>
                <th className="px-3 py-1">BBox (pixels)</th>
                <th className="px-3 py-1">Conf</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-50">
              {(rep?.elements ?? []).map((element) => (
                <ElementRow
                  key={element.id}
                  element={element}
                  imageSize={imageSize}
                  onHover={onHover}
                />
              ))}
            </tbody>
          </table>
        </div>
      </details>

      <details className="px-4 py-3 border-b border-gray-100">
        <summary className="cursor-pointer text-sm font-medium text-gray-700">
          Candidate Relationships ({rep?.candidate_relationships?.length ?? 0})
        </summary>
        <div className="mt-2 max-h-72 overflow-y-auto custom-scrollbar">
          {(rep?.candidate_relationships ?? []).map((candidate, idx) => (
            <div key={idx} className="text-xs py-0.5">
              <span className="font-mono font-semibold text-red-500">
                {candidate.label_id}
              </span>
              <span className="text-gray-400"> → </span>
              {candidate.candidates.map((c, cIdx) => (
                <span key={cIdx} className="mr-3">
                  <span className="font-mono text-green-600">{c.value_id}</span>
                  <span className="text-gray-400"> ({c.score})</span>
                </span>
              ))}
            </div>
          ))}
        </div>
      </details>

      {llmOutput && (
        <details className="px-4 py-3">
          <summary className="cursor-pointer text-sm font-medium text-gray-700">
            Raw LLM Output (JSON)
          </summary>
          <pre className="mt-2 p-3 bg-gray-900 text-gray-100 rounded text-[11px] overflow-x-auto max-h-96 custom-scrollbar">
            {JSON.stringify(llmOutput, null, 2)}
          </pre>
        </details>
      )}
    </div>
  );
}
