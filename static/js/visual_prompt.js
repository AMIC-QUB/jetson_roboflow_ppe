let canvas, ctx, resultCanvas, resultCtx;
let frameImg = new Image();
let drawing = false;
let startX, startY;
let bboxes = [];
let classes = [];

function initCanvas() {
  canvas = document.getElementById('frame-canvas');
  ctx = canvas.getContext('2d');
  resultCanvas = document.getElementById('result-canvas');
  resultCtx = resultCanvas.getContext('2d');

  // Ensure the canvas exists before proceeding
  if (!canvas) {
    console.error("Canvas element not found");
    document.getElementById('warning').textContent = "Canvas element not found.";
    document.getElementById('warning').style.display = "block";
    return;
  }

  // Use the global frameBase64 variable
  frameImg.src = "data:image/jpeg;base64," + window.frameBase64;
  frameImg.onload = function() {
    console.log("Frame image loaded successfully, setting canvas size");
    canvas.width = frameImg.width;
    canvas.height = frameImg.height;
    resultCanvas.width = frameImg.width;
    resultCanvas.height = frameImg.height;
    drawFrame();

    // Set up mouse events for drawing after the canvas is ready
    console.log("Attaching mouse event listeners to canvas");
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
  };
  frameImg.onerror = function() {
    console.error("Failed to load frame image");
    document.getElementById('warning').textContent = "Failed to load frame image.";
    document.getElementById('warning').style.display = "block";
  };

  // Populate class select dropdown
  const classSelect = document.getElementById('class-select');
  userPrompts.forEach((prompt, index) => {
    const option = document.createElement('option');
    option.value = index;
    option.text = prompt;
    classSelect.appendChild(option);
  });
}

function drawFrame() {
  ctx.drawImage(frameImg, 0, 0);
  // Draw existing bounding boxes
  bboxes.forEach((bbox, index) => {
    const classIndex = classes[index];
    const label = userPrompts[classIndex];
    const color = getColorForClass(label);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);
    ctx.font = '20px Arial';
    ctx.fillStyle = color;
    ctx.fillText(label, bbox[0], bbox[1] - 10);
  });
}

function getColorForClass(label) {
  const POSITIVE_CLASSES = ["worker with helmet", "safety vest"];
  const NEGATIVE_CLASSES = ["worker without helmet"];
  if (POSITIVE_CLASSES.includes(label)) return 'rgb(0, 255, 0)';
  if (NEGATIVE_CLASSES.includes(label)) return 'rgb(255, 0, 0)';
  // Generate a random color for other classes
  if (!(label in class_colors)) {
    class_colors[label] = `rgb(${Math.random() * 255}, ${Math.random() * 255}, ${Math.random() * 255})`;
  }
  return class_colors[label];
}

let class_colors = {};

function startDrawing(e) {
  console.log("Mouse down event triggered", e);
  if (e.button !== 0) return; // Only left mouse button
  drawing = true;
  const rect = canvas.getBoundingClientRect();
  startX = e.clientX - rect.left;
  startY = e.clientY - rect.top;
  console.log("Drawing started at", startX, startY);
}

function draw(e) {
  if (!drawing) return;
  console.log("Mouse move event triggered", e);
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  drawFrame(); // Redraw the frame and existing boxes
  ctx.strokeStyle = 'yellow';
  ctx.lineWidth = 2;
  ctx.strokeRect(startX, startY, x - startX, y - startY);
}

function stopDrawing(e) {
  console.log("Mouse up or out event triggered", e);
  if (!drawing) return;
  drawing = false;
  const rect = canvas.getBoundingClientRect();
  const endX = e.clientX - rect.left;
  const endY = e.clientY - rect.top;

  // Ensure the bounding box has a minimum size
  if (Math.abs(endX - startX) > 5 && Math.abs(endY - startY) > 5) {
    const x1 = Math.min(startX, endX);
    const y1 = Math.min(startY, endY);
    const x2 = Math.max(startX, endX);
    const y2 = Math.max(startY, endY);
    bboxes.push([x1, y1, x2, y2]);
    const classIndex = parseInt(document.getElementById('class-select').value);
    classes.push(classIndex);
    console.log("Bounding box added:", [x1, y1, x2, y2], "Class:", classIndex);
    drawFrame();
  } else {
    console.log("Bounding box too small, ignoring");
    drawFrame();
  }
}

function addBox() {
  // Enable drawing mode (already handled by mouse events)
  alert("Click and drag on the image to draw a bounding box. Select a class from the dropdown.");
}

function submitPrompt() {
  if (bboxes.length === 0) {
    alert("Please draw at least one bounding box.");
    return;
  }

  fetch('/process_visual_prompt', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      bboxes: bboxes,
      classes: classes
    })
  })
  .then(response => response.json())
  .then(data => {
    if (data.error) {
      document.getElementById('warning').textContent = data.error;
      document.getElementById('warning').style.display = "block";
      return;
    }

    // Hide the frame canvas and show the result canvas
    document.getElementById('frame-canvas').style.display = "none";
    document.getElementById('result-canvas').style.display = "block";

    // Draw the result frame
    const resultImg = new Image();
    resultImg.src = "data:image/jpeg;base64," + data.result_frame_base64;
    resultImg.onload = function() {
      resultCtx.drawImage(resultImg, 0, 0);
      // Draw the detections
      data.detections.forEach(det => {
        const { x1, y1, x2, y2, class: label, confidence, color, mask } = det;
        const [r, g, b] = color;

        if (mask) {
          // Draw segmentation mask
          const maskData = new Uint8ClampedArray(mask.flat());
          const maskImage = new ImageData(maskData, resultCanvas.width, resultCanvas.height);
          for (let i = 0; i < maskImage.data.length; i += 4) {
            if (maskImage.data[i] > 0) {
              maskImage.data[i] = r;     // Red
              maskImage.data[i + 1] = g; // Green
              maskImage.data[i + 2] = b; // Blue
              maskImage.data[i + 3] = 76; // Alpha (30% opacity)
            } else {
              maskImage.data[i + 3] = 0; // Transparent
            }
          }
          resultCtx.putImageData(maskImage, 0, 0);
          // Draw label above the mask
          resultCtx.font = '20px Arial';
          resultCtx.fillStyle = `rgb(${r}, ${g}, ${b})`;
          resultCtx.fillText(`${label} (${confidence.toFixed(2)})`, x1, y1 - 10);
        } else {
          // Draw bounding box
          resultCtx.strokeStyle = `rgb(${r}, ${g}, ${b})`;
          resultCtx.lineWidth = 2;
          resultCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
          // Draw label
          resultCtx.font = '20px Arial';
          resultCtx.fillStyle = `rgb(${r}, ${g}, ${b})`;
          resultCtx.fillText(`${label} (${confidence.toFixed(2)})`, x1, y1 - 10);
        }
      });
    };
  })
  .catch(error => {
    console.error("Error processing visual prompt:", error);
    document.getElementById('warning').textContent = "Failed to process visual prompt.";
    document.getElementById('warning').style.display = "block";
  });
}

function goBack() {
  window.location.href = "/";
}

window.onload = initCanvas;