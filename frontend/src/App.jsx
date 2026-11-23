import { Link, Navigate, Route, Routes, useLocation } from 'react-router-dom'
import PromptLabPage from './pages/PromptLabPage.jsx'
import './App.css'

function App() {
  const location = useLocation()

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <Link className="brand" to="/prompt-lab" aria-label="Prompt Lab home">
          <span className="brand-mark"><span /></span>
          <span>latent<span className="brand-period">.</span></span>
        </Link>

        <div className="workspace-label">PROMPT ENGINEERING</div>
        <nav className="primary-nav" aria-label="Main navigation">
          <Link
            className={`nav-link ${['/', '/prompt-lab'].includes(location.pathname) ? 'active' : ''}`}
            to="/prompt-lab"
          >
            <span className="nav-icon">⌘</span> Prompt Lab
          </Link>
        </nav>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumbs">
            <span>Prompt Engineering</span><b>/</b>
            <strong>Prompt Lab</strong>
          </div>
        </header>

        <Routes>
          <Route path="/" element={<PromptLabPage />} />
          <Route path="/prompt-lab" element={<PromptLabPage />} />
          <Route path="*" element={<Navigate replace to="/prompt-lab" />} />
        </Routes>
      </main>
    </div>
  )
}

export default App
