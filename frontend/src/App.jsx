import { useState, useEffect, useCallback, useRef } from 'react'
import { ROUTES, navigate, useRoute } from './router.js'
import Directory from './views/Directory.jsx'
import AddPerson from './views/AddPerson.jsx'
import Identify from './views/Identify.jsx'

const TOAST_LIFETIME = 4000
const TOAST_FADE = 150

const NAV = [
  { path: ROUTES.DIRECTORY, label: 'People', hint: 'Who the system knows' },
  { path: ROUTES.ADD, label: 'Add someone', hint: 'Teach it a new face' },
  { path: ROUTES.IDENTIFY, label: 'Identify', hint: 'Ask who this is' },
]

export default function App() {
  const route = useRoute()

  const [people, setPeople] = useState([])
  const [loadingPeople, setLoadingPeople] = useState(true)
  const [online, setOnline] = useState(null)
  const [config, setConfig] = useState(null)
  const [toasts, setToasts] = useState([])

  /*
   * Set when Identify finds a face it cannot name and the user chooses to
   * enrol it. Carrying the actual File across the route turns a dead end
   * ("Not recognised") into the next step, which is the whole point of
   * separating the screens instead of stacking them in one dashboard.
   */
  const [handoffPhoto, setHandoffPhoto] = useState(null)

  const timersRef = useRef(new Map())
  const nextIdRef = useRef(0)

  // ------------------------------------------------------------- toasts
  const dismissToast = useCallback((id) => {
    const timers = timersRef.current.get(id)
    if (timers) {
      clearTimeout(timers.fade)
      clearTimeout(timers.remove)
      timersRef.current.delete(id)
    }
    setToasts(prev => prev.map(t => (t.id === id ? { ...t, leaving: true } : t)))
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), TOAST_FADE)
  }, [])

  const notify = useCallback((message, tone = 'success') => {
    const id = nextIdRef.current++
    setToasts(prev => [...prev, { id, message, tone, leaving: false }])
    const fade = setTimeout(() => {
      setToasts(prev => prev.map(t => (t.id === id ? { ...t, leaving: true } : t)))
    }, TOAST_LIFETIME)
    const remove = setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id))
      timersRef.current.delete(id)
    }, TOAST_LIFETIME + TOAST_FADE)
    timersRef.current.set(id, { fade, remove })
  }, [])

  useEffect(() => {
    const timers = timersRef.current
    return () => {
      timers.forEach(({ fade, remove }) => { clearTimeout(fade); clearTimeout(remove) })
      timers.clear()
    }
  }, [])

  // -------------------------------------------------------------- data
  const loadPeople = useCallback(async () => {
    try {
      const res = await fetch('/api/persons', { headers: { Accept: 'application/json' } })
      if (!res.ok) throw new Error(`Server responded ${res.status}`)
      const data = await res.json()
      setPeople(data.persons ?? [])
      setOnline(true)
    } catch {
      setOnline(false)
    } finally {
      setLoadingPeople(false)
    }
  }, [])

  useEffect(() => { loadPeople() }, [loadPeople])

  // A public demo empties its database on restart. Saying so is the difference
  // between "this lost my data" and "this is a demo".
  useEffect(() => {
    let cancelled = false
    fetch('/api/config')
      .then(r => (r.ok ? r.json() : null))
      .then(cfg => { if (!cancelled && cfg) setConfig(cfg) })
      .catch(() => {})
    return () => { cancelled = true }
  }, [])

  const handleRemove = useCallback(async (name) => {
    try {
      const res = await fetch('/api/remove', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({ name }),
      })
      const data = await res.json().catch(() => ({}))
      if (res.ok && data.removed) {
        setPeople(prev => prev.filter(p => p !== name))
        notify(`${name} removed.`, 'success')
        loadPeople()
      } else {
        notify(data.error ?? `Could not remove ${name}.`, 'error')
      }
    } catch (err) {
      notify(`Could not reach the server: ${err.message}`, 'error')
    }
  }, [notify, loadPeople])

  // Identify hands a face to enrolment; enrolment consumes it exactly once.
  const enrolThisFace = useCallback((file) => {
    setHandoffPhoto(file)
    navigate(ROUTES.ADD)
  }, [])

  const consumeHandoff = useCallback(() => setHandoffPhoto(null), [])

  const shared = {
    people,
    loadingPeople,
    notify,
    reload: loadPeople,
    onRemove: handleRemove,
  }

  let view
  if (route === ROUTES.ADD) {
    view = <AddPerson {...shared} handoffPhoto={handoffPhoto} onConsumeHandoff={consumeHandoff} />
  } else if (route === ROUTES.IDENTIFY) {
    view = <Identify {...shared} config={config} onEnrolThisFace={enrolThisFace} />
  } else {
    view = <Directory {...shared} />
  }

  return (
    <div className="shell">
      <nav className="rail" aria-label="Main">
        <div className="rail-brand">
          <span className="rail-mark" aria-hidden="true" />
          <span className="rail-brand-text">Face&nbsp;ID</span>
        </div>

        <ul className="rail-nav">
          {NAV.map(item => {
            const active = route === item.path
            return (
              <li key={item.path}>
                <a
                  href={`#${item.path}`}
                  className={`rail-link${active ? ' is-active' : ''}`}
                  aria-current={active ? 'page' : undefined}
                >
                  <span className="rail-label">{item.label}</span>
                  <span className="rail-hint">{item.hint}</span>
                  {item.path === ROUTES.DIRECTORY && people.length > 0 && (
                    <span className="rail-count">{people.length}</span>
                  )}
                </a>
              </li>
            )
          })}
        </ul>

        <div className="rail-foot">
          <span className={`dot is-${online === null ? 'checking' : online ? 'online' : 'offline'}`} aria-hidden="true" />
          {online === null ? 'Connecting' : online ? 'Connected' : 'Server unreachable'}
        </div>
      </nav>

      <main className="stage">
        {config?.demo && (
          <p className="demo-banner">
            Public demo — enrolled faces are erased whenever the server
            restarts, and nothing is kept.
          </p>
        )}
        {view}
      </main>

      <div className="toasts" role="status" aria-live="polite">
        {toasts.map(t => (
          <div key={t.id} className={`toast is-${t.tone}${t.leaving ? ' is-leaving' : ''}`}>
            <span className="toast-text">{t.message}</span>
            <button
              type="button"
              className="toast-close"
              onClick={() => dismissToast(t.id)}
              aria-label="Dismiss notification"
            >
              &times;
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}
