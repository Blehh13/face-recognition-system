import { useState, useEffect } from 'react'

export default function Header({ peopleCount }) {
  const [status, setStatus] = useState('checking')

  useEffect(() => {
    let cancelled = false
    fetch('/health')
      .then(res => { if (!cancelled) setStatus(res.ok ? 'online' : 'offline') })
      .catch(() => { if (!cancelled) setStatus('offline') })
    return () => { cancelled = true }
  }, [])

  const statusLabel = {
    checking: 'Checking connection',
    online: 'Connected',
    offline: 'Cannot reach the server',
  }[status]

  const peopleLabel = peopleCount === 1 ? '1 person enrolled' : `${peopleCount} people enrolled`

  return (
    <header className="header">
      <div>
        <h1 className="header-title">Face Recognition</h1>
        <p className="header-sub">{peopleLabel}</p>
      </div>

      <div className="header-status">
        <span className={`status-dot is-${status}`} aria-hidden="true" />
        {statusLabel}
      </div>
    </header>
  )
}
