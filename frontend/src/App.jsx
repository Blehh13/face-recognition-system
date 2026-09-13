import { useState, useEffect, useCallback, useRef } from 'react'
import Header from './components/Header.jsx'
import EnrollForm from './components/EnrollForm.jsx'
import PersonList from './components/PersonList.jsx'
import IdentifyPanel from './components/IdentifyPanel.jsx'
import ResultsPanel from './components/ResultsPanel.jsx'

const TOAST_LIFETIME = 4000
const TOAST_FADE = 150

export default function App() {
  const [people, setPeople] = useState([])
  const [loadingPeople, setLoadingPeople] = useState(true)
  const [results, setResults] = useState(null)
  const [toasts, setToasts] = useState([])

  const timersRef = useRef(new Map())
  const nextIdRef = useRef(0)

  const dismissToast = useCallback((id) => {
    const timers = timersRef.current.get(id)
    if (timers) {
      clearTimeout(timers.fade)
      clearTimeout(timers.remove)
      timersRef.current.delete(id)
    }
    setToasts(prev => prev.map(t => (t.id === id ? { ...t, leaving: true } : t)))
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id))
    }, TOAST_FADE)
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
      timers.forEach(({ fade, remove }) => {
        clearTimeout(fade)
        clearTimeout(remove)
      })
      timers.clear()
    }
  }, [])

  const loadPeople = useCallback(async () => {
    try {
      const res = await fetch('/api/persons', { headers: { Accept: 'application/json' } })
      if (!res.ok) throw new Error(`Server responded ${res.status}`)
      const data = await res.json()
      setPeople(data.persons ?? [])
      return true
    } catch {
      return false
    } finally {
      setLoadingPeople(false)
    }
  }, [])

  useEffect(() => { loadPeople() }, [loadPeople])

  const handleEnrolled = useCallback((payload) => {
    const { name, faces_enrolled: count, warnings = [] } = payload
    if (count > 0) {
      const label = count === 1 ? '1 photo' : `${count} photos`
      notify(`Enrolled ${name} from ${label}.`, 'success')
      warnings.forEach(w => notify(w, 'warning'))
    } else if (warnings.length > 0) {
      // The per-photo warnings say *why* ("Found 2 faces", "not a readable
      // image"). A generic summary here would hide the actionable part.
      warnings.forEach(w => notify(w, 'warning'))
    } else {
      notify(`Nothing was enrolled for ${name}.`, 'warning')
    }
    loadPeople()
  }, [notify, loadPeople])

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
        notify(`Removed ${name}.`, 'success')
        loadPeople()
      } else {
        notify(data.error ?? `Could not remove ${name}.`, 'error')
      }
    } catch (err) {
      notify(`Could not reach the server: ${err.message}`, 'error')
    }
  }, [notify, loadPeople])

  const handleIdentified = useCallback((data) => {
    setResults(data)
    const count = data.num_faces ?? 0
    if (count === 0) {
      notify('No faces found in that photo.', 'warning')
      return
    }

    const facesLabel = count === 1 ? '1 face' : `${count} faces`

    // Nothing enrolled yet: say so, rather than reporting a bare non-match.
    if (data.enrolled_count === 0) {
      notify(`Found ${facesLabel}, but nobody is enrolled yet. Add a person first.`, 'warning')
      return
    }

    const known = data.faces.filter(f => f.is_known).length
    notify(
      known > 0
        ? `Found ${facesLabel}, ${known} recognised.`
        : `Found ${facesLabel}, none recognised.`,
      known > 0 ? 'success' : 'warning',
    )
  }, [notify])

  return (
    <div className="app">
      <Header peopleCount={people.length} />

      <main className="layout">
        <div className="column">
          <PersonList
            people={people}
            loading={loadingPeople}
            onRemove={handleRemove}
          />
          <EnrollForm onEnrolled={handleEnrolled} onError={msg => notify(msg, 'error')} />
        </div>

        <div className="column">
          <IdentifyPanel
            onIdentified={handleIdentified}
            onError={msg => notify(msg, 'error')}
          />
          <ResultsPanel data={results} />
        </div>
      </main>

      <div className="toasts" role="status" aria-live="polite">
        {toasts.map(toast => (
          <div
            key={toast.id}
            className={`toast is-${toast.tone}${toast.leaving ? ' is-leaving' : ''}`}
          >
            <span className="toast-text">{toast.message}</span>
            <button
              type="button"
              className="toast-close"
              onClick={() => dismissToast(toast.id)}
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
