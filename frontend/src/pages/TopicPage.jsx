import { useEffect } from 'react'
import { Link, useParams } from 'react-router-dom'
import { topics } from '../data/topics.js'

function TopicPage({ onViewed }) {
  const { topicId } = useParams()
  const topic = topics.find((item) => item.id === topicId)
  const topicIndex = topics.findIndex((item) => item.id === topicId)

  useEffect(() => {
    if (topic) onViewed(topic.id)
  }, [onViewed, topic])

  if (!topic) {
    return (
      <div className="page-content detail-not-found">
        <span className="section-kicker">TOPIC NOT FOUND</span>
        <h1>We couldn’t find that guide.</h1>
        <Link className="primary-button" to="/">Back to the library</Link>
      </div>
    )
  }

  const previousTopic = topics[topicIndex - 1]
  const nextTopic = topics[topicIndex + 1]
  const relatedTopics = topics
    .filter((item) => item.category === topic.category && item.id !== topic.id)
    .slice(0, 3)

  return (
    <div className="page-content topic-detail-page">
      <Link className="back-to-library" to="/">
        <span aria-hidden="true">←</span> Back to all topics
      </Link>

      <section className="detail-hero">
        <div className="detail-hero-copy">
          <div className="detail-breadcrumb">
            <span>{topic.category}</span><b>/</b><span>{topic.level}</span>
          </div>
          <h1>{topic.title}</h1>
          <p>{topic.summary}</p>
          <div className="detail-meta">
            <span>{topic.minutes} min read</span>
            <span className="meta-divider" />
            <span>{topic.level}</span>
          </div>
        </div>
        <div className={`detail-topic-icon topic-icon icon-${topic.color}`} aria-hidden="true">
          {topic.icon}
        </div>
        <span className="detail-hero-decoration" aria-hidden="true">✳</span>
      </section>

      <div className="lesson-layout">
        <article className="lesson-content">
          <section className="lesson-section" id="what-is-it">
            <div className="lesson-section-heading">
              <span className="lesson-step">01</span>
              <div>
                <span className="section-kicker">START WITH THE BIG IDEA</span>
                <h2>What is it?</h2>
              </div>
            </div>
            <p className="lesson-body">{topic.lesson.intro}</p>
          </section>

          <section className="lesson-section" id="key-ideas">
            <div className="lesson-section-heading">
              <span className="lesson-step">02</span>
              <div>
                <span className="section-kicker">THE IMPORTANT PARTS</span>
                <h2>Key ideas to remember</h2>
              </div>
            </div>
            <ul className="detail-points">
              {topic.lesson.points.map((point) => (
                <li key={point}><span>✓</span><p>{point}</p></li>
              ))}
            </ul>
          </section>

          <section className="lesson-section" id="example">
            <div className="lesson-section-heading">
              <span className="lesson-step">03</span>
              <div>
                <span className="section-kicker">MAKE IT CONCRETE</span>
                <h2>A quick example</h2>
              </div>
            </div>
            <div className="detail-example">
              <span className="example-label">EXAMPLE</span>
              <p>{topic.lesson.example}</p>
              <small>{topic.lesson.note}</small>
            </div>
          </section>

          <nav className="lesson-pagination" aria-label="Navigate between guides">
            {previousTopic ? (
              <Link to={`/topics/${previousTopic.id}`}>
                <span>← Previous guide</span><strong>{previousTopic.title}</strong>
              </Link>
            ) : <span />}
            {nextTopic && (
              <Link className="next-guide" to={`/topics/${nextTopic.id}`}>
                <span>Next guide →</span><strong>{nextTopic.title}</strong>
              </Link>
            )}
          </nav>
        </article>

        <aside className="lesson-aside">
          <div className="aside-card">
            <span className="section-kicker">IN THIS GUIDE</span>
            <div className="guide-outline">
              <a href="#what-is-it">The big idea</a>
              <a href="#key-ideas">Key ideas</a>
              <a href="#example">A quick example</a>
            </div>
            <div className="aside-divider" />
            <div className="guide-time"><span aria-hidden="true">◷</span> About {topic.minutes} minutes</div>
          </div>

          {relatedTopics.length > 0 && (
            <div className="aside-card related-card">
              <span className="section-kicker">KEEP EXPLORING</span>
              <h3>Related guides</h3>
              {relatedTopics.map((relatedTopic) => (
                <Link className="related-link" key={relatedTopic.id} to={`/topics/${relatedTopic.id}`}>
                  <span className={`related-icon icon-${relatedTopic.color}`}>{relatedTopic.icon}</span>
                  <span>{relatedTopic.title}</span>
                  <b aria-hidden="true">↗</b>
                </Link>
              ))}
            </div>
          )}
        </aside>
      </div>
    </div>
  )
}

export default TopicPage
