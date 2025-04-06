// --- Global Variables ---
let canvas, ctx, resultCanvas, resultCtx;
let frameImg = new Image();
let drawing = false;
let startX, startY;
let bboxes = []; // Stores drawn boxes [x1, y1, x2, y2]
let classes = []; // Stores selected class index for each box
let class_colors = {}; // Cache for generated class colors

// --- Initialization ---
function initCanvas() {
  canvas = document.getElementById('frame-canvas');
  ctx = canvas.getContext('2d');
  resultCanvas = document.getElementById('result-canvas');
  resultCtx = resultCanvas.getContext('2d');
  const warningDiv = document.getElementById('warning');


  if (!canvas || !resultCanvas) {
    console.error("Canvas elements not found");
    if(warningDiv) warningDiv.textContent = "Error: Canvas element(s) not found.";
    if(warningDiv) warningDiv.style.display = "block";
    return;
  }

  // Ensure frameBase64 and userPrompts are passed from Flask template
  if (typeof frameBase64 === 'undefined' || typeof userPrompts === 'undefined') {
      console.error("frameBase64 or userPrompts not found. Check Flask template.");
       if(warningDiv) warningDiv.textContent = "Error: Missing necessary data from server.";
       if(warningDiv) warningDiv.style.display = "block";
      return;
  }

  // Load the static frame provided by the server
  frameImg.src = "data:image/jpeg;base64," + frameBase64; // Use the global variable

  frameImg.onload = function() {
    console.log("Frame image loaded, setting canvas size");
    canvas.width = frameImg.width;
    canvas.height = frameImg.height;
    resultCanvas.width = frameImg.width; // Ensure result canvas matches
    resultCanvas.height = frameImg.height;
    drawFrame(); // Initial draw

    // Attach mouse listeners after image is loaded
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing); // Stop drawing if mouse leaves canvas
    console.log("Canvas event listeners attached.");
  };

  frameImg.onerror = function() {
    console.error("Failed to load frame image from base64 data");
     if(warningDiv) warningDiv.textContent = "Error: Failed to load frame image.";
     if(warningDiv) warningDiv.style.display = "block";
  };

  // --- Populate Class Select Dropdown ---
  const classSelect = document.getElementById('class-select');
  if (classSelect) {
    // Clear existing options first
    classSelect.innerHTML = '';
    userPrompts.forEach((prompt, index) => {
      const option = document.createElement('option');
      option.value = index; // Store the index as value
      option.textContent = prompt; // Display the name
      classSelect.appendChild(option);
    });
    console.log("Class dropdown populated.");
  } else {
     console.error("Class select dropdown not found.");
  }

  // Attach button listeners
  const addBoxBtn = document.getElementById('add-box-btn');
  const submitBtn = document.getElementById('submit-btn');
  const backBtn = document.getElementById('back-btn');

  if (addBoxBtn) addBoxBtn.addEventListener('click', addBox); // Keep for user guidance
  if (submitBtn) submitBtn.addEventListener('click', submitPrompt);
  if (backBtn) backBtn.addEventListener('click', goBack);

}

// --- Drawing Functions ---

function drawFrame() {
  if (!ctx || !frameImg.complete || frameImg.naturalWidth === 0) return; // Ensure image is ready
  ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear previous drawings
  ctx.drawImage(frameImg, 0, 0, canvas.width, canvas.height); // Draw the base image

  // Draw existing saved bounding boxes
  bboxes.forEach((bbox, index) => {
    if (classes.length <= index) return; // Ensure class data exists
    const classIndex = classes[index];
    const label = userPrompts[classIndex] || `Class ${classIndex}`; // Handle potential index mismatch
    const color = getColorForClass(label);

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);

    // Draw label above the box
    ctx.fillStyle = color;
    ctx.font = '14px Arial';
    ctx.fillText(label, bbox[0], bbox[1] - 5); // Adjust font size and position
  });
}

function getColorForClass(label) {
  // Consistent simple coloring based on label hash (or use predefined)
  let hash = 0;
  for (let i = 0; i < label.length; i++) {
    hash = label.charCodeAt(i) + ((hash << 5) - hash);
    hash = hash & hash; // Convert to 32bit integer
  }
  const r = (hash & 0xFF0000) >> 16;
  const g = (hash & 0x00FF00) >> 8;
  const b = hash & 0x0000FF;
  return `rgb(${r}, ${g}, ${b})`;
}


function startDrawing(e) {
  if (e.button !== 0) return; // Only left click
  const rect = canvas.getBoundingClientRect();
  startX = e.clientX - rect.left;
  startY = e.clientY - rect.top;
  drawing = true;
  console.log("Drawing started at:", startX, startY);
}

function draw(e) {
  if (!drawing) return;
  const rect = canvas.getBoundingClientRect();
  const currentX = e.clientX - rect.left;
  const currentY = e.clientY - rect.top;

  drawFrame(); // Redraw base image and existing boxes first

  // Draw the temporary rectangle being drawn
  ctx.strokeStyle = 'yellow'; // Use a distinct color for the temporary box
  ctx.lineWidth = 1;
  ctx.strokeRect(startX, startY, currentX - startX, currentY - startY);
}

function stopDrawing(e) {
  if (!drawing) return;
  drawing = false;
  const rect = canvas.getBoundingClientRect();
  const endX = e.clientX - rect.left;
  const endY = e.clientY - rect.top;

  console.log("Drawing stopped at:", endX, endY);

  // Check for minimal size to avoid accidental clicks
  if (Math.abs(endX - startX) > 5 && Math.abs(endY - startY) > 5) {
    // Ensure coordinates are ordered (x1 < x2, y1 < y2)
    const x1 = Math.min(startX, endX);
    const y1 = Math.min(startY, endY);
    const x2 = Math.max(startX, endX);
    const y2 = Math.max(startY, endY);

    // Get selected class *index*
    const classSelect = document.getElementById('class-select');
    const classIndex = parseInt(classSelect.value, 10);

    if (!isNaN(classIndex)) {
      bboxes.push([x1, y1, x2, y2]);
      classes.push(classIndex);
      console.log("Box added:", { bbox: [x1, y1, x2, y2], classIndex: classIndex });
      drawFrame(); // Redraw everything with the new box
    } else {
       console.warn("No valid class selected.");
       alert("Please select a class from the dropdown before finishing the box.");
       drawFrame(); // Redraw to remove the temporary yellow box
    }
  } else {
      console.log("Box too small, ignored.");
      drawFrame(); // Redraw to remove the temporary yellow box
  }
}

// --- Button Actions ---

function addBox() {
  // This button might not be strictly needed if drawing is intuitive,
  // but can provide user guidance.
  alert("Click and drag on the image to draw a box. Make sure a class is selected.");
}

function submitPrompt() {
  const warningDiv = document.getElementById('warning');
  if (bboxes.length === 0) {
    alert("Please draw at least one bounding box before submitting.");
    return;
  }

  console.log("Submitting visual prompt...");
   if(warningDiv) warningDiv.textContent = "Processing visual prompt...";
   if(warningDiv) warningDiv.style.display = "block";


  // --- Prepare Payload ---
  // Ensure frameBase64 and userPrompts are available globally from the template
  const payload = {
    bboxes: bboxes,
    classes: classes,
    frame_base64: frameBase64,           // *** ADDED: The original frame data ***
    user_prompts: userPrompts            // *** ADDED: The list of class names ***
  };

  fetch('/process_visual_prompt', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload) // Send the updated payload
  })
  .then(response => {
      if (!response.ok) {
          // Try to get error message from JSON response body
          return response.json().then(errData => {
              throw new Error(errData.error || `HTTP error! status: ${response.status}`);
          }).catch(() => {
              // Fallback if response is not JSON
              throw new Error(`HTTP error! status: ${response.status}`);
          });
      }
      return response.json();
  })
  .then(data => {
    console.log("Visual prompt processed, response received:", data);
    if(warningDiv) warningDiv.style.display = "none"; // Hide processing message

    if (data.error) { // Check for application-level errors in response
      console.error("Server returned error:", data.error);
      if(warningDiv) warningDiv.textContent = "Error: " + data.error;
      if(warningDiv) warningDiv.style.display = "block";
      return;
    }

    if (!data.result_frame_base64) {
        console.error("Server response missing result_frame_base64");
         if(warningDiv) warningDiv.textContent = "Error: Server did not return a result image.";
         if(warningDiv) warningDiv.style.display = "block";
        return;
    }

    // Display the result frame (which should have annotations from the server)
    const resultDisplayDiv = document.getElementById('result-display');
    const resultInfoDiv = document.getElementById('result-info');

    const resultImg = new Image();
    resultImg.src = "data:image/jpeg;base64," + data.result_frame_base64;
    resultImg.onload = function() {
        if (!resultCtx) return;
        resultCtx.clearRect(0, 0, resultCanvas.width, resultCanvas.height);
        resultCtx.drawImage(resultImg, 0, 0, resultCanvas.width, resultCanvas.height);
        console.log("Result image drawn onto result canvas.");

        // Hide the drawing canvas, show the result area
        if (canvas) canvas.style.display = "none";
        document.getElementById('controls').style.display = "none"; // Hide drawing controls
        if (resultDisplayDiv) resultDisplayDiv.style.display = "block";

        // Display detection info (optional)
        if (resultInfoDiv) {
            if (data.detections && data.detections.length > 0) {
                resultInfoDiv.innerHTML = `<h3>Detections:</h3><ul>` +
                    data.detections.map(d => `<li>${d.class_name} (Conf: ${d.confidence.toFixed(2)})</li>`).join('') +
                    `</ul>`;
            } else {
                 resultInfoDiv.innerHTML = `<p>No objects detected with the visual prompt.</p>`;
            }
        }

    };
     resultImg.onerror = function() {
        console.error("Failed to load result image from base64 data");
        if(warningDiv) warningDiv.textContent = "Error: Failed to display result image.";
        if(warningDiv) warningDiv.style.display = "block";
    };


  })
  .catch(error => {
    console.error("Error submitting visual prompt fetch:", error);
    if(warningDiv) warningDiv.textContent = "Failed to process visual prompt: " + error.message;
    if(warningDiv) warningDiv.style.display = "block";
  });
}

function goBack() {
  window.location.href = "/"; // Navigate back to the main page
}

// --- Initialize ---
window.onload = initCanvas;