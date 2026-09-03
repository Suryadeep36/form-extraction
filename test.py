import cv2
import json
import os

# Your template JSON
template_data = """
{
  "id": "e96b3bdf-5ee5-402c-b2d0-f339e434feba",
  "name": "tmpavhhrpux",
  "source_filename": "tmpavhhrpux.png",
  "image_width": 963,
  "image_height": 1280,
  "fields": [
    {
      "label": "Name:",
      "label_norm": "name",
      "label_bbox": [0.0945, 0.21875, 0.29283, 0.2625],
      "value_bbox": [0.29283, 0.22656, 0.74143, 0.3],
      "kind": "underline",
      "confidence": 0.8
    },
    {
      "label": "Roll No :",
      "label_norm": "roll no",
      "label_bbox": [0.09346, 0.32422, 0.32814, 0.37031],
      "value_bbox": [0.50883, 0.33398, 0.95742, 0.40742],
      "kind": "underline",
      "confidence": 0.8
    },
    {
      "label": "University:",
      "label_norm": "university",
      "label_bbox": [0.0945, 0.43047, 0.40187, 0.48203],
      "value_bbox": [0.44341, 0.43047, 0.87435, 0.48203],
      "kind": "box",
      "confidence": 0.7
    }
  ]
}
"""

def visualize_bboxes(image_path, json_data):
    # Load JSON
    data = json.loads(json_data)
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        return

    # Read image
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    # Loop through fields and draw the value_bbox
    for field in data.get("fields", []):
        label = field.get("label_norm", "unknown")
        bbox = field.get("value_bbox")
        
        # Denormalize coordinates
        x1, y1, x2, y2 = [
            int(bbox[0] * w),
            int(bbox[1] * h),
            int(bbox[2] * w),
            int(bbox[3] * h)
        ]
        
        # Draw the rectangle (Green, thickness 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label text above the box
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Save output and try to display it
    output_path = "test-suite/debug_fill2_bboxes.png"
    cv2.imwrite(output_path, img)
    print(f"Visualized image saved to: {output_path}")

    # Display the image (will skip if running in a headless environment)
    try:
        cv2.imshow("Value Bounding Boxes", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not display image (likely a headless environment). Check the saved file.")

# Assuming the image is a .png, adjust if it's .jpg
visualize_bboxes("test-suite/fill2.png", template_data)