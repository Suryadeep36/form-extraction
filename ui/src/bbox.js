export const toPct = (bbox) => {
  if (!bbox || bbox.length !== 4) return null;
  const [x1, y1, x2, y2] = bbox;
  return {
    left: `${x1 * 100}%`,
    top: `${y1 * 100}%`,
    width: `${(x2 - x1) * 100}%`,
    height: `${(y2 - y1) * 100}%`,
  };
};

export const normPixelBbox = (bbox, width, height) => {
  if (!bbox || bbox.length !== 4 || !width || !height) return null;
  return [
    bbox[0] / width,
    bbox[1] / height,
    bbox[2] / width,
    bbox[3] / height,
  ];
};

export const confLabel = (confidence) =>
  confidence == null ? 'n/a' : `${(confidence * 100).toFixed(0)}%`;

export const confBadgeClass = (confidence) => {
  if (confidence == null) return 'bg-gray-100 text-gray-600';
  if (confidence >= 0.9) return 'bg-green-100 text-green-700';
  if (confidence >= 0.7) return 'bg-amber-100 text-amber-700';
  return 'bg-red-100 text-red-700';
};

export const COLORS = {
  section: '#3b82f6',
  label: '#ef4444',
  value: '#22c55e',
  checkbox: '#a855f7',
  question: '#06b6d4',
  unassigned: '#f97316',
  element: '#94a3b8',
};
