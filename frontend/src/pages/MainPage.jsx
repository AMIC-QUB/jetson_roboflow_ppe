import React, { useState, useEffect, useRef } from 'react';
import apiService from '../services/apiService'; // Helper for API calls

function MainPage() {
  const [userPrompts, setUserPrompts] = useState('');
  const [isPaused, setIsPaused] = useState(false); // You might get initial state via API
  const [isUsingVideo, setIsUsingVideo] = useState(false);
  const [visualPromptsActive, setVisualPromptsActive] = useState(false); // Track VP status
  const [videoSource, setVideoSource] = useState('/video_feed'); // Initial source
  const [warning, setWarning] = useState('');
  const videoFeedRef = useRef(null); // Ref for the img tag

  // Fetch initial state if needed
  // useEffect(() => {
  //   apiService.getStatus().then(data => {
  //      setIsPaused(data.is_paused);
  //      setUserPrompts(data.user_prompts.join(', '));
  //      setVisualPromptsActive(data.visual_prompts_active);
  //      setIsUsingVideo(data.using_video);
  //   }).catch(err => setWarning(`Error fetching initial state: ${err.message}`));
  // }, []);

  const handleUpdatePrompts = async () => {
    setWarning('');
    try {
      const promptsArray = userPrompts.split(',').map(p => p.trim()).filter(Boolean);
      if (promptsArray.length === 0) {
        alert('Please enter prompts.');
        return;
      }
      const data = await apiService.updatePrompts(promptsArray);
      alert('Prompts updated!');
      // Maybe update state based on response if needed
    } catch (error) {
      setWarning(`Error updating prompts: ${error.message}`);
    }
  };

  const handleClearPrompts = async () => {
     setWarning('');
     try {
        const data = await apiService.clearPrompts();
        setUserPrompts('');
        setIsPaused(data.paused); // Update pause state from response
        alert(data.paused ? 'Prompts cleared. Inference paused.' : 'Prompts cleared. Inference resumed.');
        // Handle warning display based on data.paused
     } catch (error) {
        setWarning(`Error clearing prompts: ${error.message}`);
     }
  };

   const handleTogglePause = async () => {
     setWarning('');
     try {
        // Assuming you add a dedicated '/toggle_pause' endpoint in Flask
        const data = await apiService.togglePause();
        setIsPaused(data.paused);
        alert(data.paused ? 'Inference paused.' : 'Inference resumed.');
     } catch (error) {
        setWarning(`Error toggling pause: ${error.message}`);
     }
  };


  const handleClearVisualPrompts = async () => {
     setWarning('');
     try {
        const data = await apiService.clearVisualPrompts();
        setVisualPromptsActive(false); // Update state
        alert('Visual prompts cleared.');
     } catch (error) {
        setWarning(`Error clearing visual prompts: ${error.message}`);
     }
  };


  const handleVideoSelection = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    setWarning('Uploading video...');
    try {
        await apiService.uploadVideo(file);
        setIsUsingVideo(true);
        // Force reload of the video feed img source
        setVideoSource(`/video_feed?timestamp=${new Date().getTime()}`);
        setWarning('');
        alert('Video uploaded successfully.');
    } catch(error) {
        setWarning(`Error uploading video: ${error.message}`);
    }
  };

  const handleUseWebcam = async () => {
     setWarning('');
     try {
        await apiService.useWebcam(); // Make GET request to '/' or dedicated endpoint
        setIsUsingVideo(false);
        // Force reload of the video feed img source
        setVideoSource(`/video_feed?timestamp=${new Date().getTime()}`);
     } catch(error) {
        setWarning(`Error switching to webcam: ${error.message}`);
     }
  };


  // Effect to force reload image on source change
  useEffect(() => {
      if (videoFeedRef.current) {
          videoFeedRef.current.src = videoSource;
      }
  }, [videoSource]);


  return (
    <div>
      {/* Video Source Controls */}
       <div className="controls video-source-controls">
         <label htmlFor="video-input-react">Use Video File:</label>
         {/* Use a button to trigger the hidden file input */}
         <input
             type="file"
             id="video-input-react"
             accept="video/*"
             style={{ display: 'none' }}
             onChange={handleVideoSelection}
          />
         <button onClick={() => document.getElementById('video-input-react').click()} disabled={isUsingVideo}>
             Select Video File
         </button>
         <button onClick={handleUseWebcam} disabled={!isUsingVideo}>
             Switch to Webcam
         </button>
       </div>


      {/* Video Display */}
      <div className="container">
        {/* Add timestamp to prevent caching when source *type* changes */}
        <img
           id="video-feed"
           ref={videoFeedRef}
           src={videoSource} // Use state variable
           alt="Video Feed"
           onError={() => setWarning('Error loading video stream.')}
         />
      </div>

      {/* Inference Controls */}
      <div className="controls inference-controls">
        <input
          type="text"
          placeholder="Enter text prompts (comma-separated)"
          value={userPrompts}
          onChange={(e) => setUserPrompts(e.target.value)}
        />
        <button onClick={handleUpdatePrompts}>Update Text Prompts</button>
        <button onClick={handleClearPrompts}>Clear Prompts & Toggle Pause</button>
        <button onClick={handleTogglePause}>{isPaused ? 'Resume Inference' : 'Pause Inference'}</button>

         {/* Link to Visual Prompt Page */}
         {/* <Link to="/visual-prompt">Set Visual Prompts</Link> */}
         {/* OR keep button style if preferred */}
         <button onClick={() => window.location.href='/visual-prompt'}>Set Visual Prompts</button>


        {visualPromptsActive && (
          <button onClick={handleClearVisualPrompts}>Clear Visual Prompts</button>
        )}
        {visualPromptsActive && <span> Visual Prompts Active</span>}
      </div>

      {/* Warning Area */}
      {warning && <div className="warning-area">{warning}</div>}
    </div>
  );
}

export default MainPage;