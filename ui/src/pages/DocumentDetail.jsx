import { useEffect, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import Viewer from '../components/Viewer.jsx';
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

export default function DocumentDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [record, setRecord] = useState(null);
  const [error, setError] = useState(null);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      setError(null);
      setRecord(null);
      try {
        const res = await fetch(`${API_BASE}/documents/${id}`);
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
        if (!cancelled) setRecord(json);
      } catch (err) {
        console.error('Failed to load document:', err);
        if (!cancelled) setError(err.message || 'Failed to load document.');
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [id]);

  const handleDelete = async () => {
    if (!window.confirm(`Delete "${record?.original_filename}"?`)) return;
    setDeleting(true);
    try {
      const res = await fetch(`${API_BASE}/documents/${id}`, {
        method: 'DELETE',
      });
      if (!res.ok) throw new Error(`Delete failed with status ${res.status}`);
      navigate('/');
    } catch (err) {
      console.error('Failed to delete document:', err);
      setError(err.message || 'Failed to delete document.');
      setDeleting(false);
    }
  };

  if (error) {
    return (
      <div className="max-w-7xl mx-auto p-6">
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
          <span className="font-bold">Error: </span> {error}
        </div>
        <Link
          to="/"
          className="inline-block mt-4 text-sm font-semibold text-blue-600 hover:text-blue-800"
        >
          ← Back to dashboard
        </Link>
      </div>
    );
  }

  if (!record) {
    return (
      <div className="max-w-7xl mx-auto p-6">
        <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-10 text-center text-gray-500 text-sm">
          Loading document...
        </div>
      </div>
    );
  }

  const imageSrc = `${API_BASE}/documents/${id}/image`;

  return (
    <div className="max-w-7xl mx-auto p-6">
      <header className="mb-6 flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <Link
              to="/"
              className="text-sm font-semibold text-blue-600 hover:text-blue-800"
            >
              ← Dashboard
            </Link>
            <h1 className="text-2xl font-bold text-gray-900 break-all">
              {record.original_filename}
            </h1>
          </div>
          <p className="text-gray-500 mt-1 text-sm">
            Stored {formatDate(record.created_at)} ·{' '}
            {record.image_width ?? '—'} × {record.image_height ?? '—'} px
            {record.document_type ? ` · ${record.document_type}` : ''}
          </p>
        </div>
        <button
          onClick={handleDelete}
          disabled={deleting}
          className="inline-flex items-center justify-center px-4 py-2 rounded-md text-sm font-semibold text-red-700 bg-white border border-red-200 hover:bg-red-50 disabled:opacity-40 shadow-sm"
        >
          {deleting ? 'Deleting…' : 'Delete document'}
        </button>
      </header>

      {record.full_response ? (
        <Viewer
          response={record.full_response}
          imageSrc={imageSrc}
        />
      ) : (
        <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-10 text-center text-gray-500 text-sm">
          This record has no stored extraction output.
        </div>
      )}
    </div>
  );
}