function Stat({ label, value, accent }) {
  return (
    <div className="px-3 py-2 rounded-md bg-gray-50 border border-gray-100 text-center">
      <div className={`text-lg font-bold ${accent || 'text-gray-800'}`}>
        {value}
      </div>
      <div className="text-[10px] uppercase tracking-wide text-gray-400">
        {label}
      </div>
    </div>
  );
}

export default function StatsBar({ response }) {
  const data = response?.data;
  const rep = response?.document_representation;
  const pre = response?.preprocessing;

  const items = [
    { label: 'OCR Items', value: response?.ocr_items ?? '—' },
    { label: 'Regions', value: response?.detected_regions ?? '—' },
    { label: 'Checkboxes', value: response?.detected_checkboxes ?? '—' },
    {
      label: 'H/V Lines',
      value: rep
        ? `${rep.lines?.horizontal?.length ?? 0} / ${rep.lines?.vertical?.length ?? 0}`
        : '—',
    },
    {
      label: 'Candidates',
      value: rep?.candidate_relationships?.length ?? '—',
    },
    {
      label: 'Orientation',
      value: pre ? `${pre.orientation_corrected}°` : '—',
    },
    { label: 'Deskew', value: pre ? `${pre.deskew_angle}°` : '—' },
    {
      label: 'Sections',
      value: data?.sections?.length ?? '—',
    },
  ];

  return (
    <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
        <div className="min-w-[160px]">
          <div className="text-sm font-semibold text-gray-800 truncate">
            {response?.filename || '—'}
          </div>
          <div className="text-xs text-gray-500">
            {response?.image
              ? `${response.image.width} × ${response.image.height} px`
              : '—'}
          </div>
        </div>

        <div className="flex-1" />

        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">Document confidence</span>
          <span
            className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-bold ${
              !data?.document_confidence
                ? 'bg-gray-100 text-gray-600'
                : data.document_confidence >= 0.9
                  ? 'bg-green-100 text-green-700'
                  : data.document_confidence >= 0.7
                    ? 'bg-amber-100 text-amber-700'
                    : 'bg-red-100 text-red-700'
            }`}
          >
            {data?.document_confidence == null
              ? 'n/a'
              : `${(data.document_confidence * 100).toFixed(1)}%`}
          </span>
        </div>
      </div>

      <div className="mt-3 grid grid-cols-4 md:grid-cols-8 gap-2">
        {items.map((item) => (
          <Stat key={item.label} {...item} />
        ))}
      </div>
    </div>
  );
}
