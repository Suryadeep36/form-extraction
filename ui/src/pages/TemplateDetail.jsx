import { useEffect, useState, useRef, useLayoutEffect } from "react";
import { Link, useParams } from "react-router-dom";
import { API_BASE } from "../api.js";

function formatDate(iso) {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  } catch {
    return iso;
  }
}

/* ------------------------------------------------------------------ */
/*  Overlay box — positioned via px offsets relative to the actual     */
/*  rendered image box (measured with getBoundingClientRect), so it     */
/*  aligns no matter how the image is scaled/capped.                   */
/* ------------------------------------------------------------------ */

function OverlayBox({
  bbox,
  color = "#3b82f6",
  fill = "0.12",
  label,
  z = 10,
  frame,
}) {
  if (!bbox || bbox.length !== 4 || !frame) return null;
  const [x1, y1, x2, y2] = bbox;
  const left = frame.ox + x1 * frame.w;
  const top = frame.oy + y1 * frame.h;
  const style = {
    left: left,
    top: top,
    width: (x2 - x1) * frame.w,
    height: (y2 - y1) * frame.h,
  };
  return (
    <div
      className="absolute rounded-sm pointer-events-none transition-opacity duration-100"
      style={{
        ...style,
        border: `2px solid ${color}`,
        backgroundColor: color + hexAlpha(fill),
        zIndex: z,
      }}
    >
      {label && (
        <span
          className="absolute -top-5 left-0 text-[10px] font-semibold text-white px-1 py-0.5 rounded shadow-md whitespace-nowrap"
          style={{ backgroundColor: color, zIndex: z + 1 }}
        >
          {label}
        </span>
      )}
    </div>
  );
}

function hexAlpha(a) {
  return Math.round(a * 255)
    .toString(16)
    .padStart(2, "0");
}

/* ------------------------------------------------------------------ */
/*  Field card — highlights on hover                                   */
/* ------------------------------------------------------------------ */

const CHECKED_COLOR = "#22c55e";
const UNCHECKED_COLOR = "#94a3b8";
const GROUP_COLOR = "#8b5cf6";

function isCheckboxGroup(field) {
  return (
    field?.value_type === "checkbox_group" || Array.isArray(field?.options)
  );
}

function OptionPill({ option, showState }) {
  const checked = showState && option.checked === true;
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2 py-1 rounded text-xs font-medium ${
        checked
          ? "bg-green-100 text-green-800 ring-1 ring-green-300"
          : "bg-gray-100 text-gray-600"
      }`}
    >
      {showState && (
        <input
          type="checkbox"
          checked={checked}
          readOnly
          className="w-3.5 h-3.5 rounded border-gray-300 pointer-events-none"
        />
      )}
      {option.text || option.label || (
        <span className="italic text-gray-400">—</span>
      )}
    </span>
  );
}

/* Checkbox-group field: a set of option checkboxes whose extracted value is
   the list of checked option texts (single- or multi-select). */
function CheckboxGroupCard({ field, onHover, onLeave }) {
  const options = field.options || [];
  const checkedOptions = options.filter((o) => o.checked === true);
  const boxes = [];
  if (field.bbox && field.bbox.length === 4) {
    boxes.push({ bbox: field.bbox, color: GROUP_COLOR });
  }
  for (const o of options) {
    if (o.bbox && o.bbox.length === 4) {
      boxes.push({
        bbox: o.bbox,
        color: o.checked === true ? CHECKED_COLOR : UNCHECKED_COLOR,
      });
    }
  }
  return (
    <div
      className="bg-gray-50 rounded p-2 text-sm border border-gray-100 cursor-pointer hover:border-purple-400 hover:shadow-sm transition-all duration-100"
      onMouseEnter={() =>
        boxes.length && onHover?.({ label: field.label, boxes })
      }
      onMouseLeave={() => onLeave?.()}
    >
      <div className="flex items-start justify-between gap-2 mb-1.5">
        <span className="block text-gray-500 text-xs break-words">
          {field.label || (
            <span className="italic text-gray-400">unlabelled</span>
          )}
        </span>
        <div className="flex items-center gap-1 shrink-0">
          <span className="text-[9px] uppercase tracking-wide bg-purple-100 text-purple-700 px-1 py-0.5 rounded">
            checkbox group
          </span>
          <span className="text-[10px] text-gray-400">
            {checkedOptions.length}/{options.length}
          </span>
        </div>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {options.map((option, idx) => (
          <OptionPill key={idx} option={option} showState />
        ))}
        {options.length === 0 && (
          <span className="text-xs text-gray-400 italic">No options</span>
        )}
      </div>
      {checkedOptions.length > 0 && (
        <div className="mt-1.5 text-xs text-gray-600">
          Selected:{" "}
          <span className="font-semibold text-green-700">
            {checkedOptions.map((o) => o.text).join(", ")}
          </span>
        </div>
      )}
    </div>
  );
}

function FieldCard({ field, onHover, onLeave }) {
  if (isCheckboxGroup(field)) {
    return (
      <CheckboxGroupCard field={field} onHover={onHover} onLeave={onLeave} />
    );
  }
  const hasBbox = field.bbox && field.bbox.length === 4;
  return (
    <div
      className={`bg-gray-50 rounded p-2 text-sm border transition-all duration-100 ${
        hasBbox
          ? "cursor-pointer hover:border-blue-400 hover:shadow-sm"
          : "border-gray-100"
      }`}
      onMouseEnter={() =>
        hasBbox &&
        onHover?.({
          label: field.label,
          boxes: [{ bbox: field.bbox, color: "#22c55e" }],
        })
      }
      onMouseLeave={() => onLeave?.()}
    >
      <span className="block text-gray-500 text-xs mb-1 break-words">
        {field.label}
      </span>
      <div className="flex items-center gap-2">
        <span className="font-medium break-words text-gray-900">
          {field.value ?? <span className="italic text-gray-400">null</span>}
        </span>
        {field.value_type && field.value_type !== "text" && (
          <span className="text-[9px] uppercase tracking-wide bg-gray-200 text-gray-500 px-1 py-0.5 rounded">
            {field.value_type}
          </span>
        )}
        {field.source && (
          <span className="text-[9px] uppercase tracking-wide bg-gray-100 text-gray-400 px-1 py-0.5 rounded">
            {field.source}
          </span>
        )}
        {field.confidence != null && (
          <span className="text-[10px] text-gray-400 ml-auto">
            {(field.confidence * 100).toFixed(0)}%
          </span>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Table view — cells highlight on hover                              */
/* ------------------------------------------------------------------ */

function TableView({ table, onHover, onLeave }) {
  if (!table.cells || table.cells.length === 0) {
    return <p className="text-sm text-gray-400 italic">No cells.</p>;
  }

  const rows = table.n_rows || 0;
  const columns = table.n_cols || 0;
  const grid = Array.from({ length: rows }, () =>
    Array.from({ length: columns }, () => null),
  );

  for (const cell of table.cells) {
    if (cell.row == null || cell.column == null) continue;
    const row = cell.row;
    const col = cell.column;
    for (let r = row; r < row + (cell.row_span || 1); r++) {
      for (let c = col; c < col + (cell.column_span || 1); c++) {
        if (grid[r] && grid[r][c] !== undefined) grid[r][c] = cell;
      }
    }
  }

  return (
    <div className="overflow-x-auto border border-gray-200 rounded">
      <table className="min-w-full divide-y divide-gray-200 text-sm">
        <tbody className="divide-y divide-gray-200 bg-white">
          {grid.map((rowCells, rIdx) => (
            <tr key={rIdx} className="divide-x divide-gray-200">
              {rowCells.map((cell, cIdx) => {
                const hasBbox = cell?.bbox && cell.bbox.length === 4;
                return (
                  <td
                    key={cIdx}
                    className={`pl-2 pr-1 py-1 align-top transition-colors duration-75 ${
                      hasBbox ? "cursor-pointer hover:bg-blue-50" : ""
                    } ${cell ? "" : "bg-gray-50"}`}
                    onMouseEnter={() =>
                      hasBbox &&
                      onHover?.({
                        label: `r${cell.row}c${cell.column}${cell.text ? `: ${cell.text}` : ""}`,
                        boxes: [{ bbox: cell.bbox, color: "#3b82f6" }],
                      })
                    }
                    onMouseLeave={() => onLeave?.()}
                  >
                    {cell ? (
                      <>
                        <div className="whitespace-nowrap">
                          {cell.text ?? (
                            <span className="italic text-gray-400">null</span>
                          )}
                        </div>
                        <div className="text-[9px] text-gray-300 mt-0.5">
                          r{cell.row}c{cell.column}
                          {cell.row_span > 1 || cell.column_span > 1
                            ? ` (${cell.row_span}x${cell.column_span})`
                            : ""}
                        </div>
                      </>
                    ) : null}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Main page                                                          */
/* ------------------------------------------------------------------ */

export default function TemplateDetail() {
  const { id } = useParams();
  const [template, setTemplate] = useState(null);
  const [error, setError] = useState(null);
  const [loadingExtract, setLoadingExtract] = useState(false);
  const [result, setResult] = useState(null);
  const [hovered, setHovered] = useState(null);
  const [frame, setFrame] = useState(null);
  const fileRef = useRef(null);
  const imgBoxRef = useRef(null);
  const wrapBoxRef = useRef(null);

  // Measure the actually-rendered image box (offset + size within its
  // wrapper) whenever a new result arrives or the pointer lands on a field.
  useLayoutEffect(() => {
    const im = imgBoxRef.current;
    const wr = wrapBoxRef.current;
    if (!im || !wr) return;
    const ir = im.getBoundingClientRect();
    const w = wr.getBoundingClientRect();
    if (!ir.width || !ir.height) return;
    setFrame({
      ox: ir.left - w.left,
      oy: ir.top - w.top,
      w: ir.width,
      h: ir.height,
    });
  }, [result, hovered]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setError(null);
      try {
        const res = await fetch(`${API_BASE}/templates/${id}`);
        if (!res.ok)
          throw new Error(`Server responded with status: ${res.status}`);
        const json = await res.json();
        if (!cancelled) setTemplate(json.template || null);
      } catch (err) {
        console.error("Failed to load template:", err);
        if (!cancelled) setError(err.message || "Failed to load template.");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [id]);

  const handleExtract = async () => {
    const file = fileRef.current?.files?.[0];
    if (!file) return;

    setLoadingExtract(true);
    setError(null);
    setResult(null);
    try {
      const formData = new FormData();
      formData.append("image", file);
      formData.append("template_id", id);

      const res = await fetch(`${API_BASE}/extract-filled`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        let detail = `Server responded with status: ${res.status}`;
        try {
          const body = await res.json();
          if (body.detail) detail = body.detail;
        } catch {
          // ignore
        }
        throw new Error(detail);
      }
      const json = await res.json();
      setResult(json.extraction);
    } catch (err) {
      console.error("Failed to extract filled form:", err);
      setError(err.message || "Failed to extract filled form.");
    } finally {
      setLoadingExtract(false);
    }
  };

  if (template === null && !error) {
    return (
      <div className="max-w-7xl mx-auto p-6">
        <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-10 text-center text-gray-500 text-sm">
          Loading template...
        </div>
      </div>
    );
  }

  if (template === null) {
    return (
      <div className="max-w-7xl mx-auto p-6">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700 text-sm">
          <span className="font-bold">Error: </span> {error}
        </div>
        <Link
          to="/templates"
          className="inline-block mt-4 text-sm font-semibold text-blue-600 hover:text-blue-800"
        >
          ← Back to templates
        </Link>
      </div>
    );
  }

  const fields = result?.fields || null;
  const tables = result?.tables || null;

  return (
    <div className="max-w-7xl mx-auto p-6">
      <header className="mb-6 flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <Link
            to="/templates"
            className="text-sm font-semibold text-blue-600 hover:text-blue-800"
          >
            ← All templates
          </Link>
          <h1 className="text-3xl font-bold text-gray-900 mt-1">
            {template.name}
          </h1>
          <p className="text-gray-600 mt-1 text-sm">
            {template.source_filename} · {template.field_count} field(s)
            {template.table_count > 0
              ? ` · ${template.table_count} table(s)`
              : ""}
          </p>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Sticky preview: aligned filled form (hover overlay) + blank template */}
        <div>
          <div className="lg:sticky lg:top-6 flex flex-col gap-4">
            {result?.warped_image_data_url && (
              <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
                <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-2">
                  Aligned Filled Form
                </h3>
                <div className="rounded border border-gray-100 overflow-hidden bg-gray-50 text-center">
                  <div
                    ref={wrapBoxRef}
                    className="relative inline-block align-top"
                  >
                    <img
                      ref={imgBoxRef}
                      src={result.warped_image_data_url}
                      alt="Aligned filled form"
                      className="max-h-[60vh] w-auto block"
                    />
                    {/* Bbox overlay layer — shown when hovering fields / cells */}
                    {hovered && hovered.boxes && frame && (
                      <div className="absolute inset-0 pointer-events-none">
                        {hovered.boxes.map((box, idx) => (
                          <OverlayBox
                            key={idx}
                            bbox={box.bbox}
                            color={box.color}
                            fill={0.15}
                            label={idx === 0 ? hovered.label : null}
                            frame={frame}
                          />
                        ))}
                      </div>
                    )}
                  </div>
                </div>
                <p className="text-[10px] text-gray-400 mt-2 italic">
                  Hover over fields or table cells in the results to highlight
                  their location.
                </p>
              </div>
            )}

            <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
              <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-3">
                Blank Form Template
              </h3>
              <img
                src={`${API_BASE}/templates/${id}/image`}
                alt={template.name}
                className="w-full rounded border border-gray-100"
              />
              <p className="text-xs text-gray-400 mt-2">
                Created {formatDate(template.created_at)}
              </p>
              {template.fields?.some(isCheckboxGroup) && (
                <div className="mt-3 border-t border-gray-100 pt-3">
                  <h4 className="text-[10px] font-bold text-gray-400 uppercase tracking-wider mb-2">
                    Detected Checkbox Groups
                  </h4>
                  <div className="flex flex-col gap-2">
                    {template.fields.filter(isCheckboxGroup).map((f, idx) => (
                      <div key={idx} className="text-xs">
                        <span className="block text-gray-500 mb-1 break-words">
                          {f.label || (
                            <span className="italic text-gray-400">
                              unlabelled
                            </span>
                          )}
                        </span>
                        <div className="flex flex-wrap gap-1">
                          {(f.options || []).map((o, oIdx) => (
                            <OptionPill key={oIdx} option={o} showState={false} />
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Upload filled form + results */}
        <div className="lg:col-span-2 flex flex-col gap-4">
          <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Upload Filled Form
            </label>
            <div className="flex flex-col sm:flex-row gap-3 items-start sm:items-center">
              <input
                ref={fileRef}
                type="file"
                accept="image/*"
                className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
              />
              <button
                onClick={handleExtract}
                // disabled={!fileRef.current?.files?.length || loadingExtract}
                className={`flex items-center justify-center px-6 py-2.5 rounded-md font-semibold text-white transition-all whitespace-nowrap ${
                  !fileRef.current?.files?.length || loadingExtract
                    ? "bg-blue-300 cursor-not-allowed"
                    : "bg-blue-600 hover:bg-blue-700 shadow-md"
                }`}
              >
                {loadingExtract ? "Extracting…" : "Extract Key / Values"}
              </button>
            </div>
            <p className="text-xs text-gray-400 mt-2">
              The filled copy is aligned to the template and each field's value
              is read by OCR.
            </p>
          </div>

          {error && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
              <span className="font-bold">Error: </span> {error}
            </div>
          )}

          {result && (
            <div className="flex flex-col gap-4">
              {result.alignment?.method && (
                <div className="text-xs text-gray-500">
                  Alignment:{" "}
                  <span className="font-semibold">
                    {result.alignment.method}
                  </span>
                </div>
              )}

              {/* ---- Key / Values ---- */}
              {fields && fields.length > 0 && (
                <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wider">
                      Key / Values
                    </h3>
                    <span className="text-xs text-gray-400">
                      {fields.length} field(s)
                    </span>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {fields.map((field, idx) => (
                      <FieldCard
                        key={idx}
                        field={field}
                        onHover={setHovered}
                        onLeave={() => setHovered(null)}
                      />
                    ))}
                  </div>
                </div>
              )}

              {/* ---- Tables ---- */}
              {tables && tables.length > 0 && (
                <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200 flex flex-col gap-4">
                  <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wider">
                    Tables
                  </h3>
                  {tables.map((table, idx) => (
                    <div key={idx} className="flex flex-col gap-1">
                      <div className="text-xs text-gray-500">
                        {table.id} · {table.n_rows}×{table.n_cols} ·{" "}
                        {table.structure_source || "unknown"} structure
                      </div>
                      <TableView
                        table={table}
                        onHover={setHovered}
                        onLeave={() => setHovered(null)}
                      />
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
