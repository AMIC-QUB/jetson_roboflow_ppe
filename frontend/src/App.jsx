import React, { useState, useEffect } from 'react';
import { Routes, Route, Link } from 'react-router-dom';
import MainPage from './pages/MainPage';
import VisualPromptPage from './pages/VisualPromptPage';
import './App.css'; // App-specific styles

function App() {
  // You might have some global state here if needed (e.g., using Context API or Zustand)
  // For simplicity, state is managed within pages for now

  return (
    <div className="App">
      <h1>AMIC Construction Monitoring (React Frontend)</h1>
      <nav>
        {/* Basic navigation example */}
        <Link to="/">Main Feed</Link> | <Link to="/visual-prompt">Set Visual Prompt</Link>
      </nav>
      <Routes>
        <Route path="/" element={<MainPage />} />
        <Route path="/visual-prompt" element={<VisualPromptPage />} />
        {/* Add other routes if needed */}
      </Routes>
    </div>
  );
}

export default App;