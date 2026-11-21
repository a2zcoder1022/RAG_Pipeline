import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import { categories, topics } from './data/topics.js'
import NotFoundPage from './pages/NotFoundPage.jsx'
import TopicPage from './pages/TopicPage.jsx'
import './App.css'

function ArrowIcon() {
  return <span aria-hidden="true">↗</span>
}

function App() {
  const [query, setQuery] = useState('')
  const [selectedCategory, setSelectedCategory] = useState('All topics')
  const [exploredTopics, setExploredTopics] = useState([])
  const searchInputRef = useRef(null)
  const location = useLocation()
  const navigate = useNavigate()
  const progressPercent = Math.round((exploredTopics.length / topics.length) * 100)
  const currentTopic = topics.find((topic) => location.pathname === `/topics/${topic.id}`)

  const markTopicExplored = useCallback((topicId) => {
    setExploredTopics((current) => (
      current.includes(topicId) ? current : [...current, topicId]
    ))
  }, [])

  useEffect(() => {
    const handleShortcut = (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        searchInputRef.current?.focus()
      }
    }

    window.addEventListener('keydown', handleShortcut)
    return () => window.removeEventListener('keydown', handleShortcut)
  }, [])

  useEffect(() => {
    if (location.pathname === '/' && location.hash === '#explore') {
      document.getElementById('explore')?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [location.hash, location.pathname])

  const filteredTopics = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase()
    return topics.filter((topic) => {
      const matchesCategory =
        selectedCategory === 'All topics' || topic.category === selectedCategory
      const searchableText = `${topic.title} ${topic.category} ${topic.summary}`.toLowerCase()
      return matchesCategory && searchableText.includes(normalizedQuery)
    })
  }, [query, selectedCategory])

  const scrollToTopics = () => {
    if (location.pathname !== '/') {
      navigate('/#explore')
      return
    }
    document.getElementById('explore')?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <Link className="brand" to="/" aria-label="Latent home">
          <span className="brand-mark"><span /></span>
          <span>latent<span className="brand-period">.</span></span>
        </Link>

        <div className="workspace-label">LEARNING SPACE</div>
        <nav className="primary-nav" aria-label="Main navigation">
          <Link className={`nav-link ${location.pathname === '/' ? 'active' : ''}`} to="/">
            <span className="nav-icon">⌂</span> Overview
          </Link>
          <Link className="nav-link" to="/#explore" onClick={(event) => {
            event.preventDefault()
            scrollToTopics()
          }}>
            <span className="nav-icon">▦</span> Explore topics
            <span className="nav-count">09</span>
          </Link>
        </nav>

        <div className="sidebar-divider" />
        <div className="workspace-label topics-label">TOPIC LIBRARY</div>
        <nav className="topic-nav" aria-label="Filter topics by category">
          {categories.map((category) => (
            <button
              className={`topic-nav-link ${selectedCategory === category ? 'selected' : ''}`}
              key={category}
              onClick={() => {
                setSelectedCategory(category)
                if (location.pathname !== '/') navigate('/')
              }}
              type="button"
            >
              <span className={`category-dot dot-${category.toLowerCase().replace(' ', '-')}`} />
              {category}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="progress-card">
            <div className="progress-card-heading">
              <span className="progress-spark">✦</span>
              <span>Your learning path</span>
            </div>
            <p>You’re off to a great start. Keep exploring at your own pace.</p>
            <div className="progress-meta">
              <span>{exploredTopics.length} of {topics.length} topics explored</span>
              <span>{progressPercent}%</span>
            </div>
            <div className="progress-track"><span style={{ width: `${progressPercent}%` }} /></div>
          </div>
          <div className="profile-row">
            <div className="avatar">A</div>
            <div className="profile-copy"><strong>AI Explorer</strong><span>Curious learner</span></div>
            <button className="more-button" aria-label="More profile options" type="button">···</button>
          </div>
        </div>
      </aside>

      <main className="main-content" id="overview">
        <header className="topbar">
          <div className="breadcrumbs">
            <span>Learning space</span><b>/</b>
            <strong>{currentTopic?.title ?? 'Overview'}</strong>
          </div>
          <label className="search-box">
            <span className="search-icon" aria-hidden="true">⌕</span>
            <input
              aria-label="Search lessons"
              onChange={(event) => {
                setQuery(event.target.value)
                if (event.target.value) setSelectedCategory('All topics')
                if (location.pathname !== '/') navigate('/')
              }}
              placeholder="Search any topic..."
              ref={searchInputRef}
              type="search"
              value={query}
            />
            <kbd>⌘ K</kbd>
          </label>
          <button className="help-button" aria-label="About this learning space" type="button">?</button>
        </header>

        <Routes>
          <Route
            path="/"
            element={
              <div className="page-content">
                <section className="hero-panel">
                  <div className="hero-copy">
                    <div className="eyebrow"><span className="eyebrow-star">✳</span> YOUR FRIENDLY GUIDE TO AI</div>
                    <h1>Make sense of<br />what makes <span>AI tick.</span></h1>
                    <p className="hero-description">
                      Clear, practical guides to the ideas behind large language models. Start with the basics, follow your curiosity, and learn by exploring.
                    </p>
                    <button className="primary-button" onClick={scrollToTopics} type="button">
                      Explore the library <ArrowIcon />
                    </button>
                    <div className="hero-footnote"><span className="footnote-line" /> No jargon overload. Just the good stuff.</div>
                  </div>
                  <div className="hero-art" aria-hidden="true">
                    <div className="art-glow" />
                    <div className="orbit orbit-one" />
                    <div className="orbit orbit-two" />
                    <div className="art-node node-top">input</div>
                    <div className="art-node node-left">tokens</div>
                    <div className="art-node node-right">context</div>
                    <div className="art-core"><span>LLM</span><i>✦</i></div>
                    <div className="art-node node-bottom">ideas out</div>
                    <div className="art-caption"><span className="caption-dot" /> A little less mysterious</div>
                  </div>
                  <div className="hero-index">01 <span>/</span> 04</div>
                </section>

                <section className="welcome-strip" aria-label="Learning space highlights">
                  <div className="welcome-icon">✦</div>
                  <div className="welcome-copy">
                    <strong>Curiosity is a great place to start.</strong>
                    <span>Pick a topic, get the big idea, then dig into the details.</span>
                  </div>
                  <div className="strip-stat"><strong>09</strong><span>bite-sized guides</span></div>
                  <div className="strip-stat"><strong>~8<span>m</span></strong><span>average read</span></div>
                  <span className="strip-decoration" aria-hidden="true">✳</span>
                </section>

                <section className="topics-section" id="explore">
                  <div className="section-heading">
                    <div>
                      <div className="section-kicker">THE KNOWLEDGE LIBRARY</div>
                      <h2>Explore the building blocks</h2>
                      <p>Big ideas, made easier to get into.</p>
                    </div>
                    <div className="topic-total"><span>{filteredTopics.length.toString().padStart(2, '0')}</span> topics</div>
                  </div>

                  <div className="filter-row" role="group" aria-label="Topic category filters">
                    {categories.map((category) => (
                      <button
                        className={`filter-chip ${selectedCategory === category ? 'active' : ''}`}
                        key={category}
                        onClick={() => setSelectedCategory(category)}
                        type="button"
                      >
                        {category}
                      </button>
                    ))}
                  </div>

                  {filteredTopics.length > 0 ? (
                    <div className="topic-grid">
                      {filteredTopics.map((topic, index) => (
                        <Link
                          className="topic-card"
                          key={topic.id}
                          onClick={() => markTopicExplored(topic.id)}
                          style={{ animationDelay: `${index * 45}ms` }}
                          to={`/topics/${topic.id}`}
                        >
                          <div className="card-topline">
                            <span className={`topic-icon icon-${topic.color}`}>{topic.icon}</span>
                            <span className="card-arrow"><ArrowIcon /></span>
                          </div>
                          <div className="card-tags">
                            <span className="category-tag">{topic.category}</span>
                            <span className="read-time">{topic.minutes} min read</span>
                          </div>
                          <h3>{topic.title}</h3>
                          <p className="topic-summary">{topic.summary}</p>
                          <div className="card-footer">
                            <span>{topic.level}</span>
                            <span className="read-link">Open guide <b>→</b></span>
                          </div>
                        </Link>
                      ))}
                    </div>
                  ) : (
                    <div className="empty-state">
                      <span>⌕</span>
                      <h3>No matching topics yet</h3>
                      <p>Try a different search or category.</p>
                      <button onClick={() => { setQuery(''); setSelectedCategory('All topics') }} type="button">Clear filters</button>
                    </div>
                  )}
                </section>

                <footer className="page-footer">
                  <span><span className="footer-mark">✳</span> LATENT LEARNING</span>
                  <span>Made for the endlessly curious.</span>
                  <span>Learning, one layer at a time <b>↗</b></span>
                </footer>
              </div>
            }
          />
          <Route path="/topics/:topicId" element={<TopicPage onViewed={markTopicExplored} />} />
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </main>
    </div>
  )
}

export default App
