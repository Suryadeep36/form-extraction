import { useEffect, useState, useRef } from 'react';
import { Link, useParams } from 'react-router-dom';
import { API_BASE } from '../api.js';

function formatDate(iso) {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  } catch {
    return iso;
  }
}

function FieldCard({ field }) {
  return (
    <div className="bg-gray-50 rounded p-2 text-sm border border-gray-100">
      <span className="block text-gray-500 text-xs mb-1 break-words">
        {field.label}
      </span>
      <div className="flex items-center gap-2">
        <span className="font-medium break-words text-gray-900">
          {field.value ?? (
            <span className="italic text-gray-400">null</span>
          )}
        </span>
        {field.value_type && field.value_type !== 'text' && (
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

function TableView({ table }) {
  if (!table.cells || table.cells.length === 0) {
    return <p className="text-sm text-gray-400 italic">No cells.</p>;
  }

  const rows = table.n_rows || 0;
  const columns = table.n_cols || 0;
  const grid = Array.from({ length: rows }, () =>
    Array.from({ length: columns }, () => null)
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
              {rowCells.map((cell, cIdx) => (
                <td
                  key={cIdx}
                  className={`pl-2 pr-1 py-1 align-top ${
                    cell ? '' : 'bg-gray-50'
                  }`}
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
                          : ''}
                      </div>
                    </>
                  ) : null}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function TemplateDetail() {
  const { id } = useParams();
  const [template, setTemplate] = useState(null);
  const [error, setError] = useState(null);
  const [loadingExtract, setLoadingExtract] = useState(false);
  const [result, setResult] = useState(null);
  const fileRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setError(null);
      try {
        const res = await fetch(`${API_BASE}/templates/${id}`);
        if (!res.ok) throw new Error(`Server responded with status: ${res.status}`);
        const json = await res.json();
        if (!cancelled) setTemplate(json.template || null);
      } catch (err) {
        console.error('Failed to load template:', err);
        if (!cancelled) setError(err.message || 'Failed to load template.');
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
      formData.append('image', file);
      formData.append('template_id', id);

      const res = await fetch(`${API_BASE}/extract-filled`, {
        method: 'POST',
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
      console.error('Failed to extract filled form:', err);
      setError(err.message || 'Failed to extract filled form.');
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
        <Link to="/templates" className="inline-block mt-4 text-sm font-semibold text-blue-600 hover:text-blue-800">
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
          <h1 className="text-3xl font-bold text-gray-900 mt-1">{template.name}</h1>
          <p className="text-gray-600 mt-1 text-sm">
            {template.source_filename} · {template.field_count} field(s)
            {template.table_count > 0 ? ` · ${template.table_count} table(s)` : ''}
          </p>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Blank form preview */}
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
                    ? 'bg-blue-300 cursor-not-allowed'
                    : 'bg-blue-600 hover:bg-blue-700 shadow-md'
                }`}
              >
                {loadingExtract ? 'Extracting…' : 'Extract Key / Values'}
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
                  Alignment: <span className="font-semibold">{result.alignment.method}</span>
                </div>
              )}

              {fields && fields.length > 0 && (
                <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wider">
                      Key / Values
                    </h3>
                    <span className="text-xs text-gray-400">{fields.length} field(s)</span>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {fields.map((field, idx) => (
                      <FieldCard key={idx} field={field} />
                    ))}
                  </div>
                </div>
              )}

              {tables && tables.length > 0 && (
                <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200 flex flex-col gap-4">
                  <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wider">
                    Tables
                  </h3>
                  {tables.map((table, idx) => (
                    <div key={idx} className="flex flex-col gap-1">
                      <div className="text-xs text-gray-500">
                        {table.id} · {table.n_rows}×{table.n_cols} ·{' '}
                        {table.structure_source || 'unknown'} structure
                      </div>
                      <TableView table={table} />
                    </div>
                  ))}
                </div>
              )}

              {result.warped_image_data_url && (
                <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                  <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-3">
                    Aligned Filled Form
                  </h3>
                  <img
                    src={result.warped_image_data_url}
                    alt="Aligned filled form"
                    className="w-full rounded border border-gray-100"
                  />
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}