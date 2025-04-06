// --- Global Variables ---
let videoFeed;
// usingVideo should be passed from the Flask template correctly
// let usingVideo = false; // Default if not passed

// --- Initialization ---
function initPage() {
  console.log("Initializing index page");
  videoFeed = document.getElementById('video-feed');

  if (!videoFeed) {
    console.error("Video feed element not found!");
    return;
  }

  // --- Event Listeners for Buttons ---
  const useWebcamBtn = document.getElementById('use-webcam-btn');
  const useVideoBtn = document.getElementById('use-video-btn');
  const videoInput = document.getElementById('video-input');
  const updateBtn = document.getElementById('update-btn');
  const clearBtn = document.getElementById('clear-btn');
  const visualPromptBtn = document.getElementById('visual-prompt-btn');
  const clearVisualPromptBtn = document.getElementById('clear-visual-prompt-btn'); // Added reference

  if (useWebcamBtn) useWebcamBtn.addEventListener('click', useWebcam);
  if (useVideoBtn) useVideoBtn.addEventListener('click', () => videoInput.click()); // Trigger file input
  if (updateBtn) updateBtn.addEventListener('click', updatePrompts);
  if (clearBtn) clearBtn.addEventListener('click', clearPrompts);
  if (clearVisualPromptBtn) clearVisualPromptBtn.addEventListener('click', clearVisualPrompts); // Added listener
  // Visual prompt button likely just navigates via href, no JS needed unless adding behavior

  // --- Event Listener for Video File Selection ---
  if (videoInput) videoInput.addEventListener('change', handleVideoSelection);

  // --- Set Initial Button Visibility based on Flask variable ---
  if (typeof usingVideo !== 'undefined') { // Check if variable exists
      if (usingVideo) {
          if (useWebcamBtn) useWebcamBtn.style.display = 'inline-block';
          if (useVideoBtn) useVideoBtn.style.display = 'none';
      } else {
          if (useWebcamBtn) useWebcamBtn.style.display = 'none';
          if (useVideoBtn) useVideoBtn.style.display = 'inline-block';
      }
  } else {
      console.warn("Flask variable 'usingVideo' not found, defaulting button visibility.");
       if (useWebcamBtn) useWebcamBtn.style.display = 'none';
       if (useVideoBtn) useVideoBtn.style.display = 'inline-block';
  }


  // --- Show Clear VP Button if Active ---
  // Ensure visualPromptsActive is passed correctly from Flask template
  if (typeof visualPromptsActive !== 'undefined' && visualPromptsActive) {
    if (clearVisualPromptBtn) clearVisualPromptBtn.style.display = "inline-block";
    const visualPromptStatus = document.getElementById('visual-prompt-status');
    if (visualPromptStatus) visualPromptStatus.textContent = "Visual Prompts Active";
  } else {
     if (clearVisualPromptBtn) clearVisualPromptBtn.style.display = "none";
  }


  // --- Handle Video Feed Errors ---
  videoFeed.onerror = function() {
    console.error("Error loading video feed stream.");
    const warningDiv = document.getElementById('warning');
    if (warningDiv) {
        warningDiv.textContent = "Error: Could not load video stream. Is the server running?";
        warningDiv.style.display = "block";
    }
  };

  console.log("Index page initialized.");
}

// --- Video/Webcam Handling ---

function handleVideoSelection(event) {
  const file = event.target.files[0];
  const warningDiv = document.getElementById('warning');

  if (file) {
    console.log(`Selected video file: ${file.name}`);
    const formData = new FormData();
    formData.append('video', file);

    // Show loading indicator?
    if(warningDiv) warningDiv.textContent = "Uploading video...";
    if(warningDiv) warningDiv.style.display = "block";


    fetch('/', { // POST to index triggers video processing
      method: 'POST',
      body: formData
    })
    .then((response) => {
      if (response.ok) {
        console.log("Video upload successful, switching feed source.");
        // Update state potentially (if needed client-side beyond button visibility)
        // usingVideo = true; // Update local flag if used elsewhere
        const useWebcamBtn = document.getElementById('use-webcam-btn');
        const useVideoBtn = document.getElementById('use-video-btn');
        if (useWebcamBtn) useWebcamBtn.style.display = 'inline-block';
        if (useVideoBtn) useVideoBtn.style.display = 'none';
        if(warningDiv) warningDiv.style.display = "none"; // Hide warning/upload message

        // Reload video feed source - add timestamp to prevent caching
        // Add slight delay for server potentially restarting thread
        setTimeout(() => {
             console.log("Reloading video feed source");
             videoFeed.src = '/video_feed?' + new Date().getTime();
        }, 500); // 500ms delay

      } else {
         console.error("Video upload failed on server.");
         response.json().then(data => {
            if(warningDiv) warningDiv.textContent = data.error || 'Failed to upload video file';
            if(warningDiv) warningDiv.style.display = "block";
         }).catch(() => {
             if(warningDiv) warningDiv.textContent = 'Failed to upload video file (non-JSON response)';
             if(warningDiv) warningDiv.style.display = "block";
         });
      }
    })
    .catch(error => {
      console.error('Error during video upload fetch:', error);
      if(warningDiv) warningDiv.textContent = 'Error uploading video file: ' + error;
      if(warningDiv) warningDiv.style.display = "block";
    });
  }
}

function useWebcam() {
  console.log("Switching to webcam feed.");
  const warningDiv = document.getElementById('warning');
  // Tell the server to switch back by making a GET request
  fetch('/', { method: 'GET' })
    .then(response => {
      if (response.ok) {
        console.log("Server acknowledged switch to webcam.");
        // usingVideo = false; // Update local flag if needed
        const useWebcamBtn = document.getElementById('use-webcam-btn');
        const useVideoBtn = document.getElementById('use-video-btn');
        if (useWebcamBtn) useWebcamBtn.style.display = 'none';
        if (useVideoBtn) useVideoBtn.style.display = 'inline-block';
        if(warningDiv) warningDiv.style.display = "none";

        // Reload video feed source - add timestamp to prevent caching
         // Add slight delay for server potentially restarting thread
        setTimeout(() => {
             console.log("Reloading video feed source");
             videoFeed.src = '/video_feed?' + new Date().getTime();
        }, 500); // 500ms delay

      } else {
        console.error("Server failed to switch to webcam.");
        if(warningDiv) warningDiv.textContent = 'Failed to switch to webcam feed on server.';
        if(warningDiv) warningDiv.style.display = "block";
      }
    })
    .catch(error => {
      console.error('Error during fetch to switch to webcam:', error);
      if(warningDiv) warningDiv.textContent = 'Error switching to webcam: ' + error;
      if(warningDiv) warningDiv.style.display = "block";
    });
}

// --- Prompt Handling ---

function updatePrompts() {
  const promptInput = document.getElementById("prompt-input");
  if (!promptInput) return;
  const promptValue = promptInput.value;

  if (promptValue.trim() === "") {
    alert("Please enter at least one prompt.");
    return;
  }
  console.log(`Updating prompts to: ${promptValue}`);
  fetch(`/update_prompts/${encodeURIComponent(promptValue)}`)
    .then((response) => {
        if (!response.ok) { throw new Error(`HTTP error! status: ${response.status}`); }
        return response.json();
    })
    .then((data) => {
      if (data.status === 'success') {
        console.log("Prompts updated successfully:", data.prompts);
        alert("Prompts updated successfully!");
      } else {
         console.error("Failed to update prompts:", data.error || 'Unknown error');
         alert("Failed to update prompts: " + (data.error || 'Unknown server error'));
      }
    })
    .catch((error) => {
      console.error("Error fetching prompt update:", error);
      alert("Failed to update prompts: " + error);
    });
}

function clearPrompts() {
  console.log("Clearing prompts and toggling pause.");
  fetch('/clear_prompts')
    .then(response => {
        if (!response.ok) { throw new Error(`HTTP error! status: ${response.status}`); }
        return response.json();
    })
    .then(data => {
      if (data.status === 'success') {
          console.log("Prompts cleared, pause state:", data.paused);
          const promptInput = document.getElementById('prompt-input');
          if (promptInput) promptInput.value = ''; // Clear the input field
          const clearBtn = document.getElementById('clear-btn');
          // Assuming the button text toggle logic is desired:
          // if (clearBtn) clearBtn.textContent = data.paused ? 'Resume Inference' : 'Clear Prompts & Pause';

          alert(data.paused ? "Prompts cleared. Inference paused." : "Prompts cleared. Inference resumed.");

          // Optional: Show warning if paused without prompts
          const warningDiv = document.getElementById('warning');
          if (data.paused && (!data.prompts || data.prompts.length === 0)) {
              if(warningDiv) warningDiv.textContent = "Inference paused: Add new prompts to resume.";
              if(warningDiv) warningDiv.style.display = "block";
          } else {
               if(warningDiv) warningDiv.style.display = "none";
          }
      } else {
          console.error("Failed to clear prompts:", data.error || 'Unknown error');
          alert("Failed to clear prompts: " + (data.error || 'Unknown server error'));
      }
    })
    .catch(error => {
      console.error("Error fetching clear prompts:", error);
      alert("Failed to clear prompts: " + error);
    });
}

function clearVisualPrompts() {
  console.log("Clearing visual prompts.");
  fetch('/clear_visual_prompts')
    .then(response => {
         if (!response.ok) { throw new Error(`HTTP error! status: ${response.status}`); }
        return response.json();
    })
    .then(data => {
       if (data.status === 'success') {
          console.log("Visual prompts cleared successfully.");
          const clearVPBtn = document.getElementById('clear-visual-prompt-btn');
          const vpStatus = document.getElementById('visual-prompt-status');
          if (clearVPBtn) clearVPBtn.style.display = "none";
          if (vpStatus) vpStatus.textContent = "";
          alert("Visual prompts cleared.");
          // Optionally force a reload or refresh state if needed
       } else {
          console.error("Failed to clear visual prompts:", data.error || 'Unknown error');
          alert("Failed to clear visual prompts: " + (data.error || 'Unknown server error'));
       }
    })
    .catch(error => {
      console.error("Error fetching clear visual prompts:", error);
      alert("Failed to clear visual prompts: " + error);
    });
}


// --- Initialize on Load ---
window.onload = initPage;

// --- Removed Functions ---
// fetchDetections() - No longer needed, annotations are in the stream
// drawDetections() - No longer needed
// updateWarnings() - No longer needed (based on polled detections)
// toggleDisplayMode() - No longer applicable client-side