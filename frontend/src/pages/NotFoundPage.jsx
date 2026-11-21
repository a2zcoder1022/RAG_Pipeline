import { Link } from 'react-router-dom'

function NotFoundPage() {
  return (
    <div className="page-content detail-not-found">
      <span className="section-kicker">PAGE NOT FOUND</span>
      <h1>This page wandered off.</h1>
      <p>The link may be outdated, but the topic library is still here.</p>
      <Link className="primary-button" to="/">Back to the library</Link>
    </div>
  )
}

export default NotFoundPage
