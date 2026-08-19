import { confLabel, confBadgeClass, COLORS } from '../bbox.js';

function ConfidenceBadge({ confidence }) {
  return (
    <span
      className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold ${confBadgeClass(confidence)}`}
    >
      {confLabel(confidence)}
    </span>
  );
}

function UncertainBadge({ reason }) {
  return (
    <span
      className="inline-flex items-center text-[10px] bg-orange-100 text-orange-700 px-1.5 py-0.5 rounded font-semibold"
      title={reason || 'Uncertain'}
    >
      Uncertain
    </span>
  );
}

function SectionFields({ section, sectionIdx, onHover }) {
  if (!section.fields || section.fields.length === 0) return null;

  return (
    <div>
      <h4 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2">
        Fields
      </h4>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {section.fields.map((field, idx) => {
          const boxes = [];
          if (field.label_bbox) {
            boxes.push({ bbox: field.label_bbox, color: COLORS.label });
          }
          if (field.value_bbox) {
            boxes.push({ bbox: field.value_bbox, color: COLORS.value });
          }
          return (
            <div
              key={idx}
              className="bg-gray-50 rounded p-2 text-sm border border-gray-100 hover:border-blue-400 hover:bg-blue-50 cursor-default transition-colors"
              onMouseEnter={() =>
                onHover({
                  type: 'field',
                  key: `${sectionIdx}-f-${idx}`,
                  boxes,
                  label: field.label,
                })
              }
              onMouseLeave={() => onHover(null)}
            >
              <div className="flex justify-between items-start gap-2">
                <span className="block text-gray-500 text-xs mb-1 break-words">
                  {field.label}
                </span>
                <div className="flex items-center gap-1 shrink-0">
                  <ConfidenceBadge confidence={field.confidence} />
                  {field.uncertain && (
                    <UncertainBadge reason={field.reason} />
                  )}
                </div>
              </div>
              <div className="flex items-center gap-1.5">
                <span
                  className={`font-medium break-words ${
                    field.uncertain ? 'text-orange-600' : 'text-gray-900'
                  }`}
                >
                  {field.value || (
                    <span className="italic text-gray-400">null</span>
                  )}
                </span>
                {field.value_type && field.value_type !== 'text' && (
                  <span className="text-[9px] uppercase tracking-wide bg-gray-200 text-gray-500 px-1 py-0.5 rounded">
                    {field.value_type}
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function SectionQuestions({ section, sectionIdx, onHover }) {
  if (!section.questions || section.questions.length === 0) return null;

  return (
    <div>
      <h4 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2">
        Questions
      </h4>
      <div className="flex flex-col gap-2">
        {section.questions.map((question, idx) => {
          const boxes = (question.options || []).map((option) => ({
            bbox: option.bbox,
            color: option.selected ? COLORS.value : COLORS.question,
          }));
          return (
            <div
              key={idx}
              className="bg-white rounded p-3 text-sm border border-cyan-100 hover:border-cyan-400 transition-colors"
              onMouseEnter={() =>
                onHover({
                  type: 'question',
                  key: `${sectionIdx}-q-${idx}`,
                  boxes,
                  label: question.question,
                })
              }
              onMouseLeave={() => onHover(null)}
            >
              <div className="flex justify-between items-start gap-2">
                <span className="font-semibold text-gray-800">
                  {question.question}
                </span>
                <div className="flex items-center gap-1 shrink-0">
                  <ConfidenceBadge confidence={question.confidence} />
                  {question.uncertain && (
                    <UncertainBadge reason={question.reason} />
                  )}
                </div>
              </div>
              <div className="mt-2 flex flex-wrap gap-2">
                {(question.options || []).map((option, oIdx) => (
                  <span
                    key={oIdx}
                    className={`inline-flex items-center gap-1.5 px-2 py-1 rounded text-xs font-medium ${
                      option.selected
                        ? 'bg-green-100 text-green-800 ring-1 ring-green-300'
                        : 'bg-gray-100 text-gray-600'
                    }`}
                  >
                    {option.selected ? '✓' : '○'} {option.label}
                  </span>
                ))}
                {(!question.options || question.options.length === 0) && (
                  <span className="text-xs text-gray-400 italic">
                    No options detected
                  </span>
                )}
              </div>
              {question.answer && (
                <div className="mt-2 text-xs text-gray-600">
                  Answer:{' '}
                  <span className="font-semibold text-green-700">
                    {question.answer}
                  </span>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function SectionCheckboxes({ section, sectionIdx, onHover }) {
  if (!section.checkboxes || section.checkboxes.length === 0) return null;

  return (
    <div>
      <h4 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2">
        Options
      </h4>
      <div className="flex flex-wrap gap-2">
        {section.checkboxes.map((checkbox, idx) => (
          <div
            key={idx}
            className={`flex items-center gap-1.5 text-sm px-2 py-1 rounded border cursor-default transition-colors ${
              checkbox.checked === true
                ? 'bg-purple-50 border-purple-200 hover:border-purple-400'
                : 'bg-gray-50 border-gray-100 hover:border-gray-300'
            }`}
            onMouseEnter={() =>
              onHover({
                type: 'checkbox',
                key: `${sectionIdx}-c-${idx}`,
                boxes: checkbox.bbox
                  ? [{ bbox: checkbox.bbox, color: COLORS.checkbox }]
                  : [],
                label: checkbox.label,
              })
            }
            onMouseLeave={() => onHover(null)}
          >
            <input
              type="checkbox"
              checked={checkbox.checked === true}
              readOnly
              className="w-3.5 h-3.5 text-purple-600 rounded border-gray-300 pointer-events-none"
            />
            <span
              className={
                checkbox.checked === true
                  ? 'font-medium text-gray-900'
                  : 'text-gray-500'
              }
            >
              {checkbox.label}
            </span>
            <ConfidenceBadge confidence={checkbox.confidence} />
          </div>
        ))}
      </div>
    </div>
  );
}

function SectionTables({ section, sectionIdx, onHover }) {
  if (!section.tables || section.tables.length === 0) return null;

  return (
    <div>
      <h4 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2">
        Tables
      </h4>
      {section.tables.map((table, idx) => (
        <div
          key={idx}
          className="overflow-x-auto border border-gray-200 rounded"
          onMouseEnter={() =>
            onHover({
              type: 'table',
              key: `${sectionIdx}-t-${idx}`,
              boxes: table.bbox
                ? [{ bbox: table.bbox, color: COLORS.section }]
                : [],
              label: table.title || `Table ${idx + 1}`,
            })
          }
          onMouseLeave={() => onHover(null)}
        >
          <table className="min-w-full divide-y divide-gray-200 text-sm">
            <thead className="bg-gray-50">
              <tr>
                {table.headers?.map((header, hIdx) => (
                  <th
                    key={hIdx}
                    className="px-3 py-2 text-left font-medium text-gray-600 whitespace-nowrap"
                  >
                    {header}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 bg-white">
              {table.rows?.map((row, rIdx) => (
                <tr key={rIdx}>
                  {row.map((cell, cIdx) => (
                    <td
                      key={cIdx}
                      className="px-3 py-2 text-gray-800 whitespace-nowrap hover:bg-green-50"
                      onMouseEnter={(e) => {
                        e.stopPropagation();
                        onHover({
                          type: 'cell',
                          key: `${sectionIdx}-t-${idx}-c-${rIdx}-${cIdx}`,
                          boxes: cell.bbox
                            ? [{ bbox: cell.bbox, color: COLORS.value }]
                            : [],
                          label: cell.value,
                        });
                      }}
                      onMouseLeave={() => onHover(null)}
                    >
                      {cell.value || (
                        <span className="italic text-gray-400">null</span>
                      )}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  );
}

export default function ExtractionPanel({ data, onHover }) {
  const sections = data?.sections || [];

  return (
    <div className="flex flex-col gap-4">
      {data?.requires_human_review && (
        <div className="bg-orange-50 border border-orange-200 rounded-lg p-3 text-sm">
          <div className="font-semibold text-orange-700 mb-1">
            Needs Human Review
          </div>
          {(data.review_reasons || []).map((reason, idx) => (
            <div key={idx} className="text-orange-600 text-xs flex gap-1.5">
              <span>•</span>
              <span>{reason}</span>
            </div>
          ))}
        </div>
      )}

      {sections.map((section, idx) => (
        <div
          key={idx}
          className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden"
          onMouseEnter={() =>
            onHover({
              type: 'section',
              key: idx,
              boxes: section.bbox
                ? [{ bbox: section.bbox, color: COLORS.section }]
                : [],
              label: section.name,
            })
          }
          onMouseLeave={() => onHover(null)}
        >
          <div className="bg-gray-50 px-4 py-3 border-b border-gray-200 flex items-center justify-between gap-2">
            <h3 className="font-semibold text-gray-800">{section.name}</h3>
            <div className="flex items-center gap-2 text-[10px] text-gray-400">
              {section.region_ids?.length > 0 && (
                <span className="bg-blue-50 text-blue-600 px-1.5 py-0.5 rounded">
                  {section.region_ids.length} region(s)
                </span>
              )}
              {section.element_ids?.length > 0 && (
                <span className="bg-gray-100 px-1.5 py-0.5 rounded">
                  {section.element_ids.length} element(s)
                </span>
              )}
            </div>
          </div>

          <div className="p-4 flex flex-col gap-4">
            <SectionFields
              section={section}
              sectionIdx={idx}
              onHover={onHover}
            />
            <SectionQuestions
              section={section}
              sectionIdx={idx}
              onHover={onHover}
            />
            <SectionCheckboxes
              section={section}
              sectionIdx={idx}
              onHover={onHover}
            />
            <SectionTables
              section={section}
              sectionIdx={idx}
              onHover={onHover}
            />
          </div>
        </div>
      ))}

      {data?.unassigned_text?.length > 0 && (
        <div className="bg-white rounded-lg shadow-sm border border-orange-200 overflow-hidden">
          <div className="bg-orange-50 px-4 py-3 border-b border-orange-200 flex items-center justify-between">
            <h3 className="font-semibold text-orange-700">
              Unassigned Text ({data.unassigned_text.length})
            </h3>
          </div>
          <div className="p-4 flex flex-wrap gap-2">
            {data.unassigned_text.map((item, idx) => (
              <span
                key={idx}
                className="inline-flex items-center gap-1 text-xs bg-orange-50 text-orange-700 px-2 py-1 rounded border border-orange-100 hover:border-orange-400 cursor-default"
                onMouseEnter={() =>
                  onHover({
                    type: 'unassigned',
                    key: idx,
                    boxes: item.bbox
                      ? [{ bbox: item.bbox, color: COLORS.unassigned }]
                      : [],
                    label: item.text,
                  })
                }
                onMouseLeave={() => onHover(null)}
              >
                {item.text}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
