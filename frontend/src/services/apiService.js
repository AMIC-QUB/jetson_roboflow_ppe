// Base URL for your Flask API
// Adjust if Flask runs on a different port or domain during development/production
const API_BASE_URL = ''; // Use relative URL if Flask serves the React app

const handleResponse = async (response) => {
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: `HTTP error! status: ${response.status}` }));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }
    return response.json();
};

const apiService = {
    // --- Main Page API Calls ---
    updatePrompts: async (promptsArray) => {
        const response = await fetch(`/update_prompts/${encodeURIComponent(promptsArray.join(','))}`);
        return handleResponse(response);
    },

    clearPrompts: async () => {
        const response = await fetch(`/clear_prompts`);
        return handleResponse(response);
    },

    togglePause: async () => {
         const response = await fetch(`/toggle_pause`); // Assuming Flask endpoint exists
         return handleResponse(response);
    },

    clearVisualPrompts: async () => {
        const response = await fetch(`/clear_visual_prompts`);
        return handleResponse(response);
    },

    uploadVideo: async (file) => {
        const formData = new FormData();
        formData.append('video', file);
        // Note: Don't set Content-Type header manually for FormData, browser does it
        const response = await fetch(`/`, { // POST to root handles upload
            method: 'POST',
            body: formData,
        });
         // Upload might not return JSON on success, just check status
         if (!response.ok) {
            const errorData = await response.json().catch(() => ({ error: `HTTP error! status: ${response.status}` }));
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
         }
         return { status: 'success' }; // Return simple success object
    },

    useWebcam: async () => {
        const response = await fetch(`/`, { method: 'GET' }); // GET root handles switch
         if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
         }
        return { status: 'success' };
    },

    // Example: Get initial status (you'd need a Flask endpoint for this)
    // getStatus: async () => {
    //     const response = await fetch(`/api/status`); // Example endpoint
    //     return handleResponse(response);
    // },

    // --- Visual Prompt API Calls ---
    getVisualPromptData: async () => {
        // Assumes Flask '/visual_prompt' GET request now returns JSON data
        const response = await fetch(`/visual_prompt_data`); // ** NEW Flask endpoint needed **
        // This Flask endpoint should return { frame_base64: "...", user_prompts: [...], width: ..., height: ... }
        return handleResponse(response);
    },

    processVisualPrompt: async (payload) => {
        const response = await fetch(`/process_visual_prompt`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        return handleResponse(response);
    },
};

export default apiService;