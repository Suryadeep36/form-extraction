import { useEffect, useState, useRef, useLayoutEffect, useCallback } from "react";
import { Link, useParams } from "react-router-dom";
import { API_BASE } from "../api.js";

function formatDate(iso) {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

/* ------------------------------------------------------------------ */
/*  Colours: a distinct hue per element so a box on the image can be   */
/*  tied back to its card in the results. Each category draws from its */
/*  own palette, so it is also obvious what kind of element it is.     */
/* ------------------------------------------------------------------ */

const FIELDS_PALETTE = [
  "#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2",
  "#db2777", "#4f46e5", "#ca8a04", "#059669", "#e11d48", "#1d4ed8",
  "#7c3aed", "#0d9488", "#b91c1c", "#0369a1",
];
const GROUP_PALETTE = [
  "#7c3aed", "#a855f7", "#c026d3", "#6d28d9", "#9333ea", "#d946ef",
];
const TABLE_PALETTE = [
  "#d97706", "#ea580c", "#b45309", "#f59e0b", "#c2410c",
];

const CHECKED_COLOR = "#16a34a";
const UNCHECKED_COLOR = "#9ca3af";

const pick = (palette, i) => palette[i % palette.length];

function hexAlpha(a) {
  return Math.round(a * 255)
    .toString(16)
    .padStart(2, "0");
}

/* ------------------------------------------------------------------ */
/*  Overlay box — positioned via px offsets relative to the actual     */
/*  rendered image box (measured with getBoundingClientRect), so it    */
/*  aligns no matter how the image is scaled/capped.                   */
/* ------------------------------------------------------------------ */

function OverlayBox({
  bbox,
  color = "#3b82f6",
  fill = "0.12",
  z = 10,
  frame,
  dim = false,
  strong = false,
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
        border: `${strong ? 3 : 2}px solid ${color}`,
        backgroundColor:
          color + hexAlpha(strong ? Math.min(Number(fill) + 0.12, 0.5) : fill),
        zIndex: dim ? 1 : strong ? z + 5 : z,
        opacity: dim ? 0.35 : 1,
      }}
    />
  );
}

/* ------------------------------------------------------------------ */
/*  Field cards — coloured edge matches the overlay on the image       */
/* ------------------------------------------------------------------ */

function isCheckboxGroup(field) {
  return (
    field?.value_type === "checkbox_group" || Array.isArray(field?.options)
  );
}

function Swatch({ color }) {
  return (
    <span
      className="inline-block w-2 h-2 rounded-full ring-1 ring-black/10 shrink-0"
      style={{ backgroundColor: color }}
    />
  );
}

function OptionPill({ option, showState, color, onHover, onLeave }) {
  const checked = showState && option.checked === true;
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2 py-1 rounded text-xs font-medium transition-colors duration-75 ${
        checked
          ? "bg-green-100 text-green-800 ring-1 ring-green-300"
          : "bg-gray-100 text-gray-600"
      } ${onHover ? "cursor-pointer" : ""}`}
      onMouseEnter={onHover ? () => onHover() : undefined}
      onMouseLeave={onLeave ? () => onLeave() : undefined}
    >
      {color && <Swatch color={color} />}
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
function CheckboxGroupCard({ field, color, hoverKey, onHover, onLeave }) {
  const options = field.options || [];
  const checkedOptions = options.filter((o) => o.checked === true);
  return (
    <div
      className="bg-gray-50 rounded p-2 text-sm border border-gray-100 cursor-pointer hover:border-purple-400 hover:shadow-sm transition-all duration-100"
      style={{ borderLeft: `4px solid ${color}` }}
      onMouseEnter={() =>
        options.length && onHover?.(hoverKey, field.label)
      }
      onMouseLeave={() => onLeave?.()}
    >
      <div className="flex items-start justify-between gap-2 mb-1.5">
        <span className="flex items-center gap-1.5 text-gray-500 text-xs break-words">
          <Swatch color={color} />
          <span>
            {field.label || (
              <span className="italic text-gray-400">unlabelled</span>
            )}
          </span>
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
          <OptionPill
            key={idx}
            option={option}
            showState
            color={option.checked === true ? CHECKED_COLOR : UNCHECKED_COLOR}
            onHover={() =>
              onHover?.(`${hoverKey}-o${idx}`, `${field.label}: ${option.text}`)
            }
            onLeave={() => onLeave?.()}
          />
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

function FieldCard({ field, color, hoverKey, onHover, onLeave }) {
  if (isCheckboxGroup(field)) {
    return (
      <CheckboxGroupCard
        field={field}
        color={color}
        hoverKey={hoverKey}
        onHover={onHover}
        onLeave={onLeave}
      />
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
      style={{ borderLeft: `4px solid ${hasBbox ? color : "#e5e7eb"}` }}
      onMouseEnter={() =>
        hasBbox && onHover?.(hoverKey, field.label)
      }
      onMouseLeave={() => onLeave?.()}
    >
      <span className="flex items-center gap-1.5 text-gray-500 text-xs mb-1 break-words">
        {hasBbox && <Swatch color={color} />}
        <span>{field.label}</span>
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

function TableView({ table, color, tableKey, onHover, onLeave }) {
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
                      hasBbox ? "cursor-pointer hover:bg-amber-50" : ""
                    } ${cell ? "" : "bg-gray-50"}`}
                    onMouseEnter={() =>
                      hasBbox &&
                      onHover?.(
                        `${tableKey}-c${cell.row}-${cell.column}`,
                        `${table.id || "Table"} r${cell.row}c${cell.column}: ${
                          cell.text || "—"
                        }`,
                      )
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
/*  Overlay builder — one coloured box per element                     */
/* ------------------------------------------------------------------ */

const keyType = (key) => key[0]; // 'f' | 'g' | 't'

// Smallest box covering all the given normalized bboxes.
function unionBbox(...boxes) {
  const valid = boxes.filter((b) => b && b.length === 4);
  if (valid.length === 0) return null;
  return [
    Math.min(...valid.map((b) => b[0])),
    Math.min(...valid.map((b) => b[1])),
    Math.max(...valid.map((b) => b[2])),
    Math.max(...valid.map((b) => b[3])),
  ];
}

// A field's box must cover BOTH its name (label region) and its value, so
// the overlay makes the whole field visible without any text on the image.
function buildOverlays(fields, tables, templateFields) {
  const items = [];
  (fields || []).forEach((field, i) => {
    const tfield = templateFields?.[i];
    if (isCheckboxGroup(field)) {
      const color = pick(GROUP_PALETTE, i);
      const key = `g${i}`;
      const options = field.options || [];
      const groupBox = unionBbox(field.bbox, tfield?.label_bbox);
      if (groupBox) {
        items.push({
          key,
          bbox: groupBox,
          color,
          fill: "0.08",
          z: 5,
        });
      }
      options.forEach((o, oi) => {
        if (o.bbox && o.bbox.length === 4) {
          items.push({
            key: `${key}-o${oi}`,
            bbox: o.bbox,
            color: o.checked === true ? CHECKED_COLOR : UNCHECKED_COLOR,
            fill: o.checked === true ? "0.28" : "0.05",
            z: 6,
          });
        }
      });
    } else {
      const color = pick(FIELDS_PALETTE, i);
      const fieldBox = unionBbox(field.bbox, tfield?.label_bbox);
      if (fieldBox) {
        items.push({
          key: `f${i}`,
          bbox: fieldBox,
          color,
          fill: "0.12",
          z: 5,
        });
      }
    }
  });

  (tables || []).forEach((table, i) => {
    const color = pick(TABLE_PALETTE, i);
    const key = `t${i}`;
    if (table.bbox && table.bbox.length === 4) {
      items.push({
        key,
        bbox: table.bbox,
        color,
        fill: "0.04",
        z: 3,
      });
    }
    (table.cells || []).forEach((cell) => {
      if (cell.bbox && cell.bbox.length === 4) {
        items.push({
          key: `${key}-c${cell.row}-${cell.column}`,
          bbox: cell.bbox,
          color,
          fill: "0.10",
          z: 4,
        });
      }
    });
  });

  return items;
}

const isRelated = (activeKey, itemKey) =>
  activeKey != null &&
  (itemKey === activeKey || itemKey.startsWith(activeKey + "-"));

/* ------------------------------------------------------------------ */
/*  Main page                                                          */
/* ------------------------------------------------------------------ */

export default function TemplateDetail() {
  const { id } = useParams();
  const [template, setTemplate] = useState(null);
  const [error, setError] = useState(null);
  const [loadingExtract, setLoadingExtract] = useState(false);
  const [loadingFilled, setLoadingFilled] = useState(false);
  const [filledForms, setFilledForms] = useState([]);
  const [loadedFilledId, setLoadedFilledId] = useState(null);
  const [result, setResult] = useState(null);
  const [frame, setFrame] = useState(null);
  const [frameTick, setFrameTick] = useState(0);
  const [active, setActive] = useState(null);
  const [toShow, setToShow] = useState({ fields: true, groups: true, tables: true });
  const fileRef = useRef(null);
  const imgBoxRef = useRef(null);
  const wrapBoxRef = useRef(null);

  // Measure the actually-rendered image box (offset + size within its
  // wrapper) whenever a new result arrives or the image (re)loads.
  useLayoutEffect(() => {
    const measure = () => {
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
    };
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [result, frameTick, active]);

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

  const loadFilledForms = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/templates/${id}/filled-forms`);
      if (!res.ok) return;
      const json = await res.json();
      setFilledForms(json.filled_forms || []);
    } catch (err) {
      console.error("Failed to load saved filled forms:", err);
    }
  }, [id]);

  useEffect(() => {
    loadFilledForms();
  }, [loadFilledForms]);

  const openFilled = async (filledId) => {
    setLoadingFilled(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/filled-forms/${filledId}`);
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
      setResult(json.filled_form?.extraction || null);
      setLoadedFilledId(filledId);
    } catch (err) {
      console.error("Failed to load saved filled form:", err);
      setError(err.message || "Failed to load saved filled form.");
    } finally {
      setLoadingFilled(false);
    }
  };

  const deleteFilled = async (filledId) => {
    if (!window.confirm("Delete this saved filled form?")) return;
    try {
      const res = await fetch(`${API_BASE}/filled-forms/${filledId}`, {
        method: "DELETE",
      });
      if (!res.ok) throw new Error(`Server responded with status: ${res.status}`);
      if (loadedFilledId === filledId) {
        setResult(null);
        setLoadedFilledId(null);
      }
      await loadFilledForms();
    } catch (err) {
      console.error("Failed to delete saved filled form:", err);
      setError(err.message || "Failed to delete saved filled form.");
    }
  };

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

      // Saving happens server-side: the extraction is stored on disk, so it
      // survives a refresh and shows up in the "Saved Filled Forms" list.
      const res = await fetch(`${API_BASE}/filled-forms`, {
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
      const record = json.filled_form || {};
      setResult(record.extraction || null);
      setLoadedFilledId(record.id || null);
      await loadFilledForms();
    } catch (err) {
      console.error("Failed to extract filled form:", err);
      setError(err.message || "Failed to extract filled form.");
    } finally {
      setLoadingExtract(false);
    }
  };

  const onHover = (key, label) => setActive({ key, label });
  const onLeave = () => setActive(null);

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

  const fields = result?.fields || [];
  const tables = result?.tables || [];
  const overlayItems = result
    ? buildOverlays(fields, tables, template?.fields)
    : [];
  const visibleItems = overlayItems.filter(
    (it) =>
      (toShow.fields && keyType(it.key) === "f") ||
      (toShow.groups && keyType(it.key) === "g") ||
      (toShow.tables && keyType(it.key) === "t"),
  );

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
            {" · "}
            <span className="text-gray-500">
              {filledForms.length} saved filled form(s)
            </span>
          </p>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Sticky preview: always-on colour-coded overlay + blank template */}
        <div>
          <div className="lg:sticky lg:top-6 flex flex-col gap-4">
            {result?.warped_image_data_url && (
              <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wider">
                    Extraction Overlay
                  </h3>
                </div>

                {/* Layer toggles */}
                <div className="flex flex-wrap gap-1.5 mb-2">
                  <button
                    onClick={() =>
                      setToShow((s) => ({ ...s, fields: !s.fields }))
                    }
                    className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-full text-[11px] font-semibold border transition-colors duration-100 ${
                      toShow.fields
                        ? "border-blue-600 text-blue-700 bg-blue-50"
                        : "border-gray-200 text-gray-400 bg-white"
                    }`}
                  >
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: FIELDS_PALETTE[0] }}
                    />
                    Fields
                    <span className="text-[9px] text-gray-400">
                      {fields.filter((f) => !isCheckboxGroup(f)).length}
                    </span>
                  </button>
                  <button
                    onClick={() =>
                      setToShow((s) => ({ ...s, groups: !s.groups }))
                    }
                    className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-full text-[11px] font-semibold border transition-colors duration-100 ${
                      toShow.groups
                        ? "border-purple-600 text-purple-700 bg-purple-50"
                        : "border-gray-200 text-gray-400 bg-white"
                    }`}
                  >
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: GROUP_PALETTE[0] }}
                    />
                    Checkboxes
                    <span className="text-[9px] text-gray-400">
                      {fields.filter((f) => isCheckboxGroup(f)).length}
                    </span>
                  </button>
                  <button
                    onClick={() =>
                      setToShow((s) => ({ ...s, tables: !s.tables }))
                    }
                    className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-full text-[11px] font-semibold border transition-colors duration-100 ${
                      toShow.tables
                        ? "border-amber-600 text-amber-700 bg-amber-50"
                        : "border-gray-200 text-gray-400 bg-white"
                    }`}
                  >
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: TABLE_PALETTE[0] }}
                    />
                    Tables
                    <span className="text-[9px] text-gray-400">
                      {tables.length}
                    </span>
                  </button>
                </div>

                <div className="rounded border border-gray-100 overflow-hidden bg-gray-50 text-center">
                  <div
                    ref={wrapBoxRef}
                    className="relative inline-block align-top"
                  >
                    <img
                      ref={imgBoxRef}
                      src={result.warped_image_data_url}
                      alt="Aligned filled form"
                      onLoad={() => setFrameTick((t) => t + 1)}
                      className="max-h-[60vh] w-auto block"
                    />
                    {frame && (
                      <div className="absolute inset-0 pointer-events-none">
                        {visibleItems.map((box) => (
                          <OverlayBox
                            key={box.key}
                            bbox={box.bbox}
                            color={box.color}
                            fill={box.fill}
                            label={box.label}
                            z={box.z}
                            frame={frame}
                            dim={active ? !isRelated(active.key, box.key) : false}
                            strong={active ? isRelated(active.key, box.key) : false}
                          />
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                <div className="mt-2 text-[11px] text-gray-500 h-4 truncate">
                  {active ? (
                    <>
                      Focus:{" "}
                      <span className="font-semibold text-gray-700">
                        {active.label}
                      </span>
                    </>
                  ) : (
                    "Hover a result card to focus its boxes on the form."
                  )}
                </div>

                {/* Legend */}
                <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-gray-500">
                  <span className="inline-flex items-center gap-1">
                    <span
                      className="w-2.5 h-2.5 rounded-sm inline-block"
                      style={{ backgroundColor: FIELDS_PALETTE[0] }}
                    />
                    Field value
                  </span>
                  <span className="inline-flex items-center gap-1">
                    <span
                      className="w-2.5 h-2.5 rounded-sm inline-block"
                      style={{ backgroundColor: GROUP_PALETTE[0] }}
                    />
                    Checkbox group
                  </span>
                  <span className="inline-flex items-center gap-1">
                    <span
                      className="w-2.5 h-2.5 rounded-sm inline-block"
                      style={{ backgroundColor: CHECKED_COLOR }}
                    />
                    Checked option
                  </span>
                  <span className="inline-flex items-center gap-1">
                    <span
                      className="w-2.5 h-2.5 rounded-sm inline-block"
                      style={{ backgroundColor: UNCHECKED_COLOR }}
                    />
                    Unchecked
                  </span>
                  <span className="inline-flex items-center gap-1">
                    <span
                      className="w-2.5 h-2.5 rounded-sm inline-block"
                      style={{ backgroundColor: TABLE_PALETTE[0] }}
                    />
                    Table (+ cells)
                  </span>
                </div>
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
              The filled copy is aligned to the template, each field's value is
              read by OCR, and the result is saved automatically so you never
              have to re-upload after a refresh.
            </p>
          </div>

          {/* Saved filled forms */}
          <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wider">
                Saved Filled Forms
              </h3>
              {loadingFilled ? (
                <span className="text-xs text-gray-400">Loading…</span>
              ) : (
                <span className="text-xs text-gray-400">
                  {filledForms.length} saved
                </span>
              )}
            </div>
            {filledForms.length === 0 ? (
              <p className="text-sm text-gray-400 italic">
                No filled forms saved yet. Upload one above and it will be
                stored here automatically.
              </p>
            ) : (
              <ul className="divide-y divide-gray-100">
                {filledForms.map((ff) => (
                  <li
                    key={ff.id}
                    className="py-2 flex items-center gap-3 group"
                  >
                    <button
                      onClick={() => openFilled(ff.id)}
                      className={`flex-1 min-w-0 text-left rounded px-2 py-1.5 transition-colors ${
                        loadedFilledId === ff.id
                          ? "bg-blue-50"
                          : "hover:bg-gray-50"
                      }`}
                    >
                      <div className="flex items-center gap-2 min-w-0">
                        {loadedFilledId === ff.id && (
                          <span className="w-2 h-2 rounded-full bg-green-500 shrink-0" />
                        )}
                        <span
                          className="font-medium text-sm text-gray-800 truncate"
                          title={ff.source_filename}
                        >
                          {ff.source_filename}
                        </span>
                        <span className="text-[10px] text-gray-400 shrink-0">
                          {ff.field_count}f · {ff.table_count}t
                        </span>
                      </div>
                      <div className="text-xs text-gray-400 mt-0.5">
                        {formatDate(ff.created_at)}
                      </div>
                    </button>
                    <button
                      onClick={() => deleteFilled(ff.id)}
                      title="Delete saved filled form"
                      className="text-xs font-medium text-red-400 hover:text-red-600 opacity-0 group-hover:opacity-100 transition-opacity px-2 py-1 rounded"
                    >
                      Delete
                    </button>
                  </li>
                ))}
              </ul>
            )}
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
              {fields.length > 0 && (
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
                        color={
                          isCheckboxGroup(field)
                            ? pick(GROUP_PALETTE, idx)
                            : pick(FIELDS_PALETTE, idx)
                        }
                        hoverKey={
                          isCheckboxGroup(field) ? `g${idx}` : `f${idx}`
                        }
                        onHover={onHover}
                        onLeave={onLeave}
                      />
                    ))}
                  </div>
                </div>
              )}

              {/* ---- Tables ---- */}
              {tables.length > 0 && (
                <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200 flex flex-col gap-4">
                  <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wider">
                    Tables
                  </h3>
                  {tables.map((table, idx) => (
                    <div
                      key={idx}
                      className="flex flex-col gap-1"
                      onMouseEnter={() =>
                        onHover(`t${idx}`, `${table.id || "Table"} ‖ ${table.n_rows}×${table.n_cols}`)
                      }
                      onMouseLeave={onLeave}
                    >
                      <div className="text-xs text-gray-500">
                        {table.id} · {table.n_rows}×{table.n_cols} ·{" "}
                        {table.structure_source || "unknown"} structure
                      </div>
                      <TableView
                        table={table}
                        color={pick(TABLE_PALETTE, idx)}
                        tableKey={`t${idx}`}
                        onHover={onHover}
                        onLeave={onLeave}
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