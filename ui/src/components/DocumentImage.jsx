import { toPct, normPixelBbox, COLORS } from '../bbox.js';

function Box({ style, color, fill = '0.10', label, title, z = 0 }) {
  if (!style) return null;
  return (
    <div
      className="absolute rounded-[1px] transition-opacity duration-100 pointer-events-none"
      style={{
        ...style,
        border: `2px solid ${color}`,
        backgroundColor: color + hexAlpha(fill),
        zIndex: z,
      }}
      title={title}
    >
      {label ? (
        <span
          className="absolute -top-5 left-0 text-[10px] font-semibold text-white px-1 py-0.5 rounded shadow-md whitespace-nowrap"
          style={{ backgroundColor: color, zIndex: z + 1 }}
        >
          {label}
        </span>
      ) : null}
    </div>
  );
}

function hexAlpha(alpha) {
  return Math.round(alpha * 255)
    .toString(16)
    .padStart(2, '0');
}

export default function DocumentImage({
  imageSrc,
  sections = [],
  unassigned = [],
  elements = [],
  imageSize,
  layers,
  hovered,
}) {
  return (
    <div className="relative border-2 border-dashed border-gray-300 bg-gray-200 rounded-lg overflow-hidden shadow-inner">
      {imageSrc ? (
        <img
          src={imageSrc}
          alt="Scanned Document"
          className="w-full h-auto block object-contain"
        />
      ) : (
        <div className="min-h-[420px] flex items-center justify-center text-gray-500 text-sm p-8">
          Select an image to preview bounding boxes.
        </div>
      )}

      {imageSrc && layers.sections && (
        <div className="absolute inset-0 pointer-events-none">
          {sections.map((section, idx) => {
            const style = toPct(section.bbox);
            if (!style) return null;
            const isHovered = hovered?.type === 'section' && hovered.key === idx;
            return (
              <Box
                key={`s-${idx}`}
                style={style}
                color={COLORS.section}
                fill={isHovered ? '0.20' : '0.06'}
                title={section.name}
                z={isHovered ? 6 : 1}
              />
            );
          })}
        </div>
      )}

      {imageSrc && layers.elements && elements.length > 0 && (
        <div className="absolute inset-0 pointer-events-none">
          {elements.map((element) => {
            const style = toPct(
              normPixelBbox(element.bbox, imageSize?.width, imageSize?.height)
            );
            if (!style) return null;
            return (
              <Box
                key={element.id}
                style={style}
                color={COLORS.element}
                fill="0.04"
                title={`${element.id}: ${element.text}`}
                z={1}
              />
            );
          })}
        </div>
      )}

      {imageSrc && layers.labels && (
        <div className="absolute inset-0 pointer-events-none">
          {sections.map((section, sIdx) =>
            (section.fields || []).map((field, fIdx) => (
              <Box
                key={`l-${sIdx}-${fIdx}`}
                style={toPct(field.label_bbox)}
                color={COLORS.label}
                fill="0.03"
                title={field.label}
                z={1}
              />
            ))
          )}
        </div>
      )}

      {imageSrc && layers.values && (
        <div className="absolute inset-0 pointer-events-none">
          {sections.map((section, sIdx) =>
            (section.fields || []).map((field, fIdx) => {
              const boxes = field.value_bboxes || [field.value_bbox];
              return boxes.map((bbox, bIdx) => (
                <Box
                  key={`v-${sIdx}-${fIdx}-${bIdx}`}
                  style={toPct(bbox)}
                  color={COLORS.value}
                  fill="0.03"
                  title={field.value || field.label}
                  z={1}
                />
              ));
            })
          )}
        </div>
      )}

      {imageSrc && layers.checkboxes && (
        <div className="absolute inset-0 pointer-events-none">
          {sections.map((section, sIdx) =>
            (section.checkboxes || []).map((checkbox, cIdx) => (
              <Box
                key={`cb-${sIdx}-${cIdx}`}
                style={toPct(checkbox.bbox)}
                color={COLORS.checkbox}
                fill="0.05"
                title={checkbox.label}
                z={1}
              />
            ))
          )}
        </div>
      )}

      {imageSrc && hovered && hovered.boxes && (
        <div className="absolute inset-0 pointer-events-none">
          {hovered.boxes.map((box, idx) => (
            <Box
              key={idx}
              style={toPct(box.bbox)}
              color={box.color}
              fill="0.18"
              label={idx === 0 ? hovered.label : null}
              z={10}
            />
          ))}
        </div>
      )}

      {imageSrc && layers.unassigned && unassigned.length > 0 && (
        <div className="absolute inset-0 pointer-events-none">
          {unassigned.map((item, idx) => {
            const style = toPct(item.bbox);
            if (!style) return null;
            return (
              <Box
                key={`u-${idx}`}
                style={style}
                color={COLORS.unassigned}
                fill="0.05"
                title={item.text}
                z={1}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}
