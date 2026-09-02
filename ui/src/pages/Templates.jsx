import { useEffect, useState, useCallback, useRef } from 'react';
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

export default function Templates() {
  const [templates, setTemplates] = useState(null);
  const [error, setError] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadName, setUploadName] = useState('');
  const fileRef = useRef(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/templates`);
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
      setTemplates(json.templates || []);
    } catch (err) {
      console.error('Failed to load templates:', err);
      setError(err.message || 'Failed to load templates.');
      setTemplates([]);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const handleUpload = async () => {
    const file = fileRef.current?.files?.[0];
    if (!file) return;

    setUploading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('image', file);
      if (uploadName.trim()) formData.append('name', uploadName.trim());

      const res = await fetch(`${API_BASE}/register-template`, {
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
      await res.json();
      setUploadName('');
      if (fileRef.current) fileRef.current.value = '';
      await load();
    } catch (err) {
      console.error('Failed to register template:', err);
      setError(err.message || 'Failed to register template.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto p-6">
      <header className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900">Blank Form Templates</h1>
        <p className="text-gray-600 mt-1">
          Upload a blank form to create a template, then send filled copies of
          the same layout through it to extract key/value results.
        </p>
      </header>

      <div className="mb-8 bg-white p-6 rounded-lg shadow-sm border border-gray-200">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Upload Blank Form
        </label>
        <div className="flex flex-col sm:flex-row gap-3 items-start sm:items-center">
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
          />
          <input
            type="text"
            value={uploadName}
            onChange={(e) => setUploadName(e.target.value)}
            placeholder="Template name (optional)"
            className="w-full sm:w-64 text-sm rounded-md border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            onClick={handleUpload}
            disabled={!fileRef.current?.files?.length || uploading}
            className={`flex items-center justify-center px-6 py-2.5 rounded-md font-semibold text-white transition-all ${
              !fileRef.current?.files?.length || uploading
                ? 'bg-blue-300 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700 shadow-md'
            }`}
          >
            {uploading ? 'Registering…' : 'Register Template'}
          </button>
        </div>
        <p className="text-xs text-gray-400 mt-2">
          JPEG or PNG. The empty value regions are detected automatically.
        </p>
      </div>

      {error && (
        <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
          <span className="font-bold">Error: </span> {error}
        </div>
      )}

      {templates === null ? (
        <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-10 text-center text-gray-500 text-sm">
          Loading templates...
        </div>
      ) : templates.length === 0 ? (
        <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-10 text-center">
          <p className="text-gray-700 font-medium">No templates yet.</p>
          <p className="text-gray-500 text-sm mt-1">
            Upload a blank form above to create your first template.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {templates.map((template) => (
            <Link
              key={template.id}
              to={`/templates/${template.id}`}
              className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden flex flex-col hover:shadow-md transition-shadow"
            >
              <div className="bg-gray-100 relative">
                <img
                  src={`${API_BASE}/templates/${template.id}/image`}
                  alt={template.name}
                  loading="lazy"
                  className="w-full h-44 object-cover object-top"
                />
                <div className="absolute inset-0" />
              </div>
              <div className="p-4 flex flex-col gap-1.5 flex-1">
                <div className="font-semibold text-gray-900 truncate" title={template.name}>
                  {template.name}
                </div>
                <div className="text-xs text-gray-500">
                  {template.source_filename}
                </div>
                <div className="flex flex-wrap items-center gap-2 text-xs">
                  <span className="px-2 py-0.5 rounded-full bg-blue-50 text-blue-700 font-medium">
                    {template.field_count} field(s)
                  </span>
                  {template.table_count > 0 && (
                    <span className="px-2 py-0.5 rounded-full bg-purple-50 text-purple-700 font-medium">
                      {template.table_count} table(s)
                    </span>
                  )}
                </div>
                <div className="text-xs text-gray-400 mt-auto pt-2">
                  {formatDate(template.created_at)}
                </div>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}