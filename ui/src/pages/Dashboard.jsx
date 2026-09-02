import { useEffect, useState, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { API_BASE } from '../api.js';

function formatDate(iso) {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return iso;
  }
}

export default function Dashboard() {
  const [docs, setDocs] = useState(null);
  const [error, setError] = useState(null);
  const [deleting, setDeleting] = useState(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/documents`);
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
      setDocs(json.documents || []);
    } catch (err) {
      console.error('Failed to load documents:', err);
      setError(err.message || 'Failed to load documents.');
      setDocs([]);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  let handleDelete = async (id, filename) => {
    if (!window.confirm(`Delete "${filename}"?`)) return;
    setDeleting(id);
    try {
      const res = await fetch(`${API_BASE}/documents/${id}`, {
        method: 'DELETE',
      });
      if (!res.ok) throw new Error(`Delete failed with status ${res.status}`);
      await load();
    } catch (err) {
      console.error('Failed to delete document:', err);
      setError(err.message || 'Failed to delete document.');
    } finally {
      setDeleting(null);
    }
  };

  return (
    <div className="max-w-7xl mx-auto p-6">
      <header className="mb-6 flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Uploaded Documents</h1>
          <p className="text-gray-600 mt-1">
            Every extraction is stored with its image and key/value results.
          </p>
        </div>
        <Link
          to="/extract"
          className="inline-flex items-center justify-center px-4 py-2 rounded-md text-sm font-semibold text-white bg-blue-600 hover:bg-blue-700 shadow-sm"
        >
          New Extraction
        </Link>
      </header>

      {error && (
        <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
          <span className="font-bold">Error: </span> {error}
        </div>
      )}

      {docs === null ? (
        <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-10 text-center text-gray-500 text-sm">
          Loading documents...
        </div>
      ) : docs.length === 0 ? (
        <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-10 text-center">
          <p className="text-gray-700 font-medium">No documents yet.</p>
          <p className="text-gray-500 text-sm mt-1">
            Upload and analyze a document to see it here.
          </p>
          <Link
            to="/extract"
            className="inline-flex items-center justify-center mt-4 px-4 py-2 rounded-md text-sm font-semibold text-white bg-blue-600 hover:bg-blue-700 shadow-sm"
          >
            Start extraction
          </Link>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {docs.map((doc) => (
            <div
              key={doc.id}
              className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden flex flex-col"
            >
              <Link to={`/document/${doc.id}`} className="block bg-gray-100">
                {doc.image_width ? (
                  <img
                    src={`${API_BASE}/documents/${doc.id}/image`}
                    alt={doc.original_filename}
                    loading="lazy"
                    className="w-full h-40 object-cover object-top"
                  />
                ) : (
                  <div className="w-full h-40 flex items-center justify-center text-gray-400 text-sm">
                    No preview
                  </div>
                )}
              </Link>

              <div className="p-4 flex flex-col gap-2 flex-1">
                <div className="font-semibold text-gray-900 truncate" title={doc.original_filename}>
                  {doc.original_filename}
                </div>
                <div className="flex flex-wrap items-center gap-2 text-xs">
                  {doc.document_type ? (
                    <span className="px-2 py-0.5 rounded-full bg-blue-50 text-blue-700 font-medium capitalize">
                      {doc.document_type}
                    </span>
                  ) : (
                    <span className="px-2 py-0.5 rounded-full bg-gray-100 text-gray-500 font-medium">
                      unknown type
                    </span>
                  )}
                  {doc.image_width ? (
                    <span className="text-gray-400">
                      {doc.image_width} × {doc.image_height}
                    </span>
                  ) : null}
                </div>
                <div className="text-xs text-gray-400">{formatDate(doc.created_at)}</div>

                <div className="mt-auto pt-3 flex items-center justify-between">
                  <Link
                    to={`/document/${doc.id}`}
                    className="text-sm font-semibold text-blue-600 hover:text-blue-800"
                  >
                    Open →
                  </Link>
                  <button
                    onClick={() => handleDelete(doc.id, doc.original_filename)}
                    disabled={deleting === doc.id}
                    className="text-xs font-medium text-red-600 hover:text-red-800 disabled:opacity-40"
                  >
                    {deleting === doc.id ? 'Deleting…' : 'Delete'}
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}