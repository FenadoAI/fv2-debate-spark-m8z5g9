import "./App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import DebatePrep from "./components/DebatePrep";

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<DebatePrep />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
