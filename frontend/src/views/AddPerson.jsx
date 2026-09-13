import { useState, useRef, useEffect } from 'react'
import CameraCapture from '../components/CameraCapture.jsx'
import { ROUTES, navigate } from '../router.js'

/**
 * Guided enrolment: choose photos, name the person, confirm.
 *
 * Deliberately a sequence rather than a form. Enrolment is the step that
 * decides how well everything else works — a bad photo here degrades every
 * future match — so the photos get their own moment of attention instead of
 * sharing a panel with an unrelated task.
 */
export default function AddPerson({ notify, reload, people, handoffPhoto, onConsumeHandoff }) {
  const [files, setFiles] = useState([])
  const [previews, setPreviews] = useState([])
  const [name, setName] = useState('')
  const [dragging, setDragging] = useState(false)
  const [cameraOpen, setCameraOpen] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [done, setDone] = useState(null)
  const inputRef = useRef(null)

  // A face handed over from Identify arrives already chosen.
  useEffect(() => {
    if (handoffPhoto) {
      setFiles([handoffPhoto])
      onConsumeHandoff()
    }
  }, [handoffPhoto, onConsumeHandoff])

  useEffect(() => {
    const urls = files.map(f => URL.createObjectURL(f))
    setPreviews(urls)
    return () => urls.forEach(URL.revokeObjectURL)
  }, [files])

  const addFiles = (list) => {
    const images = Array.from(list).filter(f => f.type.startsWith('image/'))
    if (images.length === 0) {
      notify('Those files are not images.', 'error')
      return
    }
    setFiles(prev => [...prev, ...images])
  }

  const removeAt = (i) => setFiles(prev => prev.filter((_, idx) => idx !== i))

  const submit = async (event) => {
    event.preventDefault()
    const trimmed = name.trim()
    if (!trimmed) return notify('Give this person a name.', 'error')
    if (files.length === 0) return notify('Add at least one photo.', 'error')

    setSubmitting(true)
    const body = new FormData()
    body.append('name', trimmed)
    files.forEach(f => body.append('images', f))

    try {
      const res = await fetch('/enroll', {
        method: 'POST', headers: { Accept: 'application/json' }, body,
      })
      const data = await res.json().catch(() => null)
      if (!res.ok || !data) {
        notify(data?.error ?? 'Enrolling failed. Check the server is running.', 'error')
        return
      }

      if (data.faces_enrolled > 0) {
        setDone({ name: data.name, count: data.faces_enrolled, warnings: data.warnings ?? [] })
        setFiles([])
        setName('')
        reload()
      } else {
        // The per-photo warnings say *why*; a generic summary would hide it.
        const reasons = data.warnings ?? []
        if (reasons.length) reasons.forEach(w => notify(w, 'warning'))
        else notify(`Nothing was enrolled for ${data.name}.`, 'warning')
      }
    } catch (err) {
      notify(`Could not reach the server: ${err.message}`, 'error')
    } finally {
      setSubmitting(false)
    }
  }

  // ---------------------------------------------------------- confirmation
  if (done) {
    return (
      <div className="view">
        <div className="onboard">
          <p className="onboard-step is-good">Enrolled</p>
          <h1 className="onboard-title">{done.name} is in the directory</h1>
          <p className="onboard-body">
            Learned from {done.count === 1 ? 'one photo' : `${done.count} photos`}.
            {people.length <= 1
              ? ' Add more people, or try identifying a face now.'
              : ' Try identifying a photo to see it work.'}
          </p>
          {done.warnings.map((w, i) => (
            <p key={i} className="onboard-warn">{w}</p>
          ))}
          <div className="onboard-actions">
            <button type="button" className="btn btn-lg" onClick={() => navigate(ROUTES.IDENTIFY)}>
              Identify a face
            </button>
            <button type="button" className="btn btn-ghost" onClick={() => setDone(null)}>
              Add another person
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="view">
      <header className="view-head">
        <div>
          <h1 className="view-title">Add someone</h1>
          <p className="view-sub">Show the system a face, and give it a name.</p>
        </div>
      </header>

      <form onSubmit={submit} className="flow">
        <section className="step">
          <div className="step-mark">1</div>
          <div className="step-body">
            <h2 className="step-title">Photos of one person</h2>
            <p className="step-help">
              One clear, well-lit face per photo. Several angles work better than one.
              A photo with more than one face is refused, so a name never picks up a stranger.
            </p>

            {cameraOpen ? (
              <CameraCapture
                disabled={submitting}
                onCapture={f => { setCameraOpen(false); addFiles([f]) }}
                onError={msg => notify(msg, 'error')}
                onClose={() => setCameraOpen(false)}
              />
            ) : (
              <>
                <button
                  type="button"
                  className={`drop${dragging ? ' is-dragging' : ''}`}
                  disabled={submitting}
                  onClick={() => inputRef.current?.click()}
                  onDragOver={e => { e.preventDefault(); setDragging(true) }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={e => {
                    e.preventDefault(); setDragging(false)
                    if (!submitting && e.dataTransfer.files.length) addFiles(e.dataTransfer.files)
                  }}
                >
                  <span className="drop-title">Drop photos here</span>
                  <span className="drop-sub">or click to browse · JPG, PNG, WebP</span>
                </button>
                <button
                  type="button"
                  className="btn-link"
                  onClick={() => setCameraOpen(true)}
                  disabled={submitting}
                >
                  Use the camera instead
                </button>
              </>
            )}

            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              multiple
              hidden
              onChange={e => { if (e.target.files.length) addFiles(e.target.files); e.target.value = '' }}
            />

            {files.length > 0 && previews.length === files.length && (
              <ul className="shots">
                {files.map((f, i) => (
                  <li key={previews[i]} className="shot">
                    <img src={previews[i]} alt={f.name} />
                    <button
                      type="button"
                      className="shot-remove"
                      onClick={() => removeAt(i)}
                      aria-label={`Remove ${f.name}`}
                      disabled={submitting}
                    >
                      &times;
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </section>

        <section className={`step${files.length === 0 ? ' is-waiting' : ''}`}>
          <div className="step-mark">2</div>
          <div className="step-body">
            <h2 className="step-title">Their name</h2>
            <p className="step-help">This is what the system will call them when it finds a match.</p>
            <input
              className="field"
              type="text"
              value={name}
              onChange={e => setName(e.target.value)}
              placeholder="e.g. Alice Johnson"
              disabled={submitting}
              autoComplete="off"
            />
          </div>
        </section>

        <div className="flow-actions">
          <button
            type="submit"
            className="btn btn-lg"
            disabled={submitting || files.length === 0 || !name.trim()}
          >
            {submitting ? (<><span className="spinner" aria-hidden="true" />Enrolling</>) : 'Enrol this person'}
          </button>
          <button type="button" className="btn-link" onClick={() => navigate(ROUTES.DIRECTORY)}>
            Back to people
          </button>
        </div>
      </form>
    </div>
  )
}
