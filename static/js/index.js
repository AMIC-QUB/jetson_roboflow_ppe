let detections = [];
let showSegmentation = false;
let canvas, ctx;

function initCanvas() {
  const videoFeed = document.getElementById('video-feed');
  canvas = document.getElementById('overlay-canvas');
  ctx = canvas.getContext('2d');

  // Set canvas size to match video feed
  videoFeed.onload = function() {
    console.log("Video feed loaded, setting canvas size");
    canvas.width = videoFeed.width;
    canvas.height = videoFeed.height;
    drawDetections();
  };

  videoFeed.onerror = function() {
    console.error("Error loading video feed");
    document.getElementById('warning').textContent = "Error: Could not load video stream.";
    document.getElementById('warning').style.display = "block";
  };

  // Adjust canvas size if the video feed resizes
  videoFeed.addEventListener('resize', function() {
    console.log("Video feed resized, updating canvas size");
    canvas.width = videoFeed.width;
    canvas.height = videoFeed.height;
  });

  // Start polling for detections
  console.log("Starting to poll for detections");
  setInterval(fetchDetections, 100); // Poll every 100ms

  // Continuously redraw the canvas to ensure overlays stay in sync
  function redrawLoop() {
    drawDetections();
    requestAnimationFrame(redrawLoop);
  }
  redrawLoop();

  // Show the clear visual prompt button if visual prompts are active
  if (window.visualPromptsActive) {
    document.getElementById('clear-visual-prompt-btn').style.display = "inline-block";
    document.getElementById('visual-prompt-status').textContent = "Visual Prompts Active";
  }
}

function fetchDetections() {
  fetch('/detections')
    .then(response => response.json())
    .then(data => {
      detections = data.detections || [];
      console.log("Fetched detections:", detections);
      updateWarnings();
    })
    .catch(error => console.error("Error fetching detections:", error));
}

function drawDetections() {
  if (!ctx) return;

  // Clear the canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw detections
  detections.forEach(det => {
    const { x1, y1, x2, y2, class: label, confidence, color, mask } = det;
    const [r, g, b] = color;

    if (showSegmentation && mask) {
      // Draw segmentation mask
      const maskData = new Uint8ClampedArray(mask.flat());
      const maskImage = new ImageData(maskData, canvas.width, canvas.height);
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
      ctx.putImageData(maskImage, 0, 0);
      // Draw label above the mask
      ctx.font = '20px Arial';
      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.fillText(`${label} (${confidence.toFixed(2)})`, x1, y1 - 10);
    } else {
      // Draw bounding box
      ctx.strokeStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      // Draw label
      ctx.font = '20px Arial';
      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.fillText(`${label} (${confidence.toFixed(2)})`, x1, y1 - 10);
    }
  });
}

// Fetch detection data and update warning
function updateWarnings() {
  const warningDiv = document.getElementById("warning");
  const noHelmet = detections.some(d => d.class === "worker without helmet");
  const noVest = detections.some(d => d.class === "no-vest");

  if (noHelmet || noVest) {
    let warningText = "WARNING: ";
    if (noHelmet) warningText += "No helmet detected. ";
    if (noVest) warningText += "No vest detected.";
    warningDiv.textContent = warningText;
    warningDiv.style.display = "block";
  } else {
    warningDiv.style.display = "none";
  }
}

// Function to update prompts
function updatePrompts() {
  const promptInput = document.getElementById("prompt-input").value;
  if (promptInput.trim() === "") {
    alert("Please enter at least one prompt.");
    return;
  }
  fetch(`/update_prompts/${encodeURIComponent(promptInput)}`)
    .then((response) => response.json())
    .then((data) => {
      console.log("Prompts updated:", data.prompts);
      alert("Prompts updated successfully!");
    })
    .catch((error) => {
      console.error("Error updating prompts:", error);
      alert("Failed to update prompts.");
    });
}

// Function to clear prompts and toggle pause
function clearPrompts() {
  fetch('/clear_prompts')
    .then(response => response.json())
    .then(data => {
      console.log("Prompts cleared, pause state:", data.paused);
      document.getElementById('prompt-input').value = ''; // Clear the input field
      const clearBtn = document.getElementById('clear-btn');
      clearBtn.textContent = data.paused ? 'Resume Inference' : 'Clear Prompts'; // Update button text
      alert(data.paused ? "Inference paused." : "Inference resumed.");
      if (!data.paused && data.prompts.length === 0) {
        document.getElementById('warning').textContent = "Inference paused: Add new prompts to resume.";
        document.getElementById('warning').style.display = "block";
      }
    })
    .catch(error => {
      console.error("Error clearing prompts:", error);
      alert("Failed to clear prompts.");
    });
}

// Function to clear visual prompts
function clearVisualPrompts() {
  fetch('/clear_visual_prompts')
    .then(response => response.json())
    .then(data => {
      console.log("Visual prompts cleared");
      document.getElementById('clear-visual-prompt-btn').style.display = "none";
      document.getElementById('visual-prompt-status').textContent = "";
      alert("Visual prompts cleared.");
    })
    .catch(error => {
      console.error("Error clearing visual prompts:", error);
      alert("Failed to clear visual prompts.");
    });
}

// Function to toggle between bounding boxes and segmentation
function toggleDisplayMode() {
  fetch('/toggle_display_mode')
    .then(response => response.json())
    .then(data => {
      console.log("Display mode toggled, show_segmentation:", data.show_segmentation);
      showSegmentation = data.show_segmentation;
      const toggleBtn = document.getElementById('toggle-mode-btn');
      toggleBtn.textContent = data.show_segmentation ? 'Switch to Bounding Boxes' : 'Switch to Segmentation';
    })
    .catch(error => {
      console.error("Error toggling display mode:", error);
      alert("Failed to toggle display mode.");
    });
}

// Initialize the canvas when the page loads
window.onload = initCanvas;