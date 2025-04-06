import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import apiService from '../services/apiService'; // Helper for API calls

// Helper function for drawing (could be moved to a separate utility file)
const drawBox = (ctx, x1, y1, x2, y2, color, label) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    if (label) {
        ctx.fillStyle = color;
        ctx.font = '14px Arial';
        ctx.fillText(label, x1, y1 - 5);
    }
};

function VisualPromptPage() {
  const canvasRef = useRef(null);
  const resultCanvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPos, setStartPos] = useState({ x: 0, y: 0 });
  const [currentPos, setCurrentPos] = useState({ x: 0, y: 0 });
  const [boxes, setBoxes] = useState([]); // { bbox: [x1,y1,x2,y2], classIndex: number }
  const [selectedClassIndex, setSelectedClassIndex] = useState(0);
  const [frameData, setFrameData] = useState({ base64: null, width: 0, height: 0 });
  const [userPrompts, setUserPrompts] = useState([]); // Class names
  const [resultData, setResultData] = useState({ base64: null, detections: [] });
  const [viewMode, setViewMode] = useState('draw'); // 'draw' or 'result'
  const [warning, setWarning] = useState('');
  const navigate = useNavigate();

  // Fetch initial frame and prompts from server on load
  useEffect(() => {
    // You need an endpoint in Flask that returns the frame + prompts
    apiService.getVisualPromptData()
      .then(data => {
        setFrameData({ base64: data.frame_base64, width: data.width, height: data.height }); // Assume Flask sends dimensions
        setUserPrompts(data.user_prompts);
        if (data.user_prompts.length > 0) {
             setSelectedClassIndex(0); // Default to first class
        }
      })
      .catch(err => setWarning(`Error fetching initial data: ${err.message}`));
  }, []); // Empty dependency array means run once on mount


   // --- Canvas Drawing Logic ---
   const getCanvasCoordinates = (event) => {
     const canvas = canvasRef.current;
     if (!canvas) return { x: 0, y: 0 };
     const rect = canvas.getBoundingClientRect();
     return {
       x: event.clientX - rect.left,
       y: event.clientY - rect.top
     };
   };

   const handleMouseDown = (event) => {
     if (event.button !== 0 || viewMode !== 'draw') return;
     const pos = getCanvasCoordinates(event);
     setIsDrawing(true);
     setStartPos(pos);
     setCurrentPos(pos); // Initialize currentPos
   };

   const handleMouseMove = (event) => {
     if (!isDrawing || viewMode !== 'draw') return;
     const pos = getCanvasCoordinates(event);
     setCurrentPos(pos);
     // Redrawing happens in useEffect hook based on state changes
   };

   const handleMouseUp = (event) => {
     if (!isDrawing || viewMode !== 'draw') return;
     setIsDrawing(false);
     const start = startPos;
     const end = getCanvasCoordinates(event);

     // Check minimum size
     if (Math.abs(end.x - start.x) > 5 && Math.abs(end.y - start.y) > 5) {
       const newBox = {
         bbox: [
           Math.min(start.x, end.x),
           Math.min(start.y, end.y),
           Math.max(start.x, end.x),
           Math.max(start.y, end.y),
         ],
         classIndex: selectedClassIndex,
       };
       setBoxes(prevBoxes => [...prevBoxes, newBox]);
     }
     // Reset temporary drawing state if needed
     setCurrentPos({x:0, y:0});
     setStartPos({x:0, y:0});

   };

   // Draw effect hook - redraws canvas whenever relevant state changes
    useEffect(() => {
        if (viewMode !== 'draw' || !frameData.base64) return;
        const canvas = canvasRef.current;
        const ctx = canvas?.getContext('2d');
        if (!ctx) return;

        const img = new Image();
        img.onload = () => {
            // Set canvas size based on image
            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);

            // Draw existing boxes
            boxes.forEach(boxInfo => {
                const label = userPrompts[boxInfo.classIndex] || `Class ${boxInfo.classIndex}`;
                // Simple color generation - replace with better logic if needed
                const color = `hsl(${(boxInfo.classIndex * 40) % 360}, 100%, 50%)`;
                drawBox(ctx, ...boxInfo.bbox, color, label);
            });

            // Draw temporary box being drawn
            if (isDrawing && Math.abs(currentPos.x - startPos.x)>1 && Math.abs(currentPos.y - startPos.y)>1 ) {
                 drawBox(ctx, startPos.x, startPos.y, currentPos.x, currentPos.y, 'yellow', null);
            }
        };
        img.onerror = () => {
            setWarning("Error loading frame image for drawing.");
        }
        img.src = `data:image/jpeg;base64,${frameData.base64}`;

    }, [frameData.base64, boxes, isDrawing, startPos, currentPos, userPrompts, viewMode]); // Dependencies


    // Effect for drawing result canvas
     useEffect(() => {
         if (viewMode !== 'result' || !resultData.base64) return;
         const canvas = resultCanvasRef.current;
         const ctx = canvas?.getContext('2d');
         if (!ctx) return;

         const img = new Image();
         img.onload = () => {
             canvas.width = img.naturalWidth;
             canvas.height = img.naturalHeight;
             ctx.clearRect(0, 0, canvas.width, canvas.height);
             ctx.drawImage(img, 0, 0);
             console.log("Result image drawn.");
             // Note: Annotations are already on the result image from the server
         }
         img.onerror = () => {
            setWarning("Error loading result image.");
         }
         img.src = `data:image/jpeg;base64,${resultData.base64}`;

     }, [resultData.base64, viewMode]);


   // --- Button Actions ---
  const handleSubmitPrompt = async () => {
    if (boxes.length === 0) {
      alert("Please draw at least one bounding box.");
      return;
    }
    setWarning('Processing visual prompt...');
    try {
        const payload = {
            bboxes: boxes.map(b => b.bbox),
            classes: boxes.map(b => b.classIndex),
            frame_base64: frameData.base64, // Send the original frame
            user_prompts: userPrompts // Send the class names list
        };
      const responseData = await apiService.processVisualPrompt(payload);
      setResultData({ base64: responseData.result_frame_base64, detections: responseData.detections });
      setViewMode('result'); // Switch to result view
      setWarning('');
    } catch (error) {
      setWarning(`Error processing visual prompt: ${error.message}`);
    }
  };

  const handleGoBack = () => {
    navigate('/'); // Use react-router navigation
  };


  return (
    <div className="visual-prompt-container">
      {warning && <div className="warning-area">{warning}</div>}

      {/* Drawing Mode */}
      {viewMode === 'draw' && (
        <>
          <div className="controls">
            <label htmlFor="class-select-react">Select Class:</label>
            <select
              id="class-select-react"
              value={selectedClassIndex}
              onChange={(e) => setSelectedClassIndex(parseInt(e.target.value, 10))}
            >
              {userPrompts.map((prompt, index) => (
                <option key={index} value={index}>{prompt}</option>
              ))}
            </select>
            <button onClick={handleSubmitPrompt} disabled={boxes.length === 0}>
                Submit Visual Prompt ({boxes.length} boxes)
            </button>
             <button onClick={() => setBoxes([])} disabled={boxes.length === 0}>Clear Boxes</button>
            <button onClick={handleGoBack}>Back to Main Page</button>
          </div>
          <div className="canvas-container">
             <p>Click and drag on the image below. Selected class: <strong>{userPrompts[selectedClassIndex] || 'N/A'}</strong></p>
            <canvas
              ref={canvasRef}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp} // Handle leaving canvas area
              style={{ border: '1px solid black', cursor: 'crosshair' }}
            />
          </div>
        </>
      )}

      {/* Result Mode */}
      {viewMode === 'result' && (
         <div className="result-display">
           <h2>Visual Prompt Result</h2>
           <div className="canvas-container">
             <canvas ref={resultCanvasRef} style={{ border: '1px solid grey' }} />
           </div>
            <div className="result-info">
               <h3>Detections:</h3>
                {resultData.detections && resultData.detections.length > 0 ? (
                    <ul>
                       {resultData.detections.map((d, i) => (
                           <li key={i}>{d.class_name} (Conf: {d.confidence?.toFixed(2)})</li>
                       ))}
                    </ul>
                ) : (
                    <p>No objects detected with this prompt.</p>
                )}
           </div>
           <button onClick={handleGoBack}>Back to Main Page</button>
           <button onClick={() => setViewMode('draw')}>Draw More Prompts</button>
         </div>
       )}

    </div>
  );
}

export default VisualPromptPage;